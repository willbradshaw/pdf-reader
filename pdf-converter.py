#!/usr/bin/env python

import argparse
import pypdf
import requests
import markdown
import os
import openai
import sys
import tiktoken
import math
from functools import partial

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set maximum number of tokens for OpenAI API calls
model_max_tokens = {"gpt-3.5-turbo":4096,"gpt-4":8192}
model_chunk_tokens_base = {"gpt-3.5-turbo":math.ceil(4096/3),"gpt-4":math.ceil(8192/3)}
model_chunk_tokens_overlap = {"gpt-3.5-turbo":math.ceil(4096/30),"gpt-4":math.ceil(8192/30)}

# PDF to Text
def pdf_to_text(file_path, verbose=False):
    with open(file_path, 'rb') as file:
        pdf = pypdf.PdfReader(file)
        if verbose: print("Number of pages: {}".format(len(pdf.pages)))
        text = ""
        for page in pdf.pages: text += page.extract_text() + "\n"
        if verbose: print("Text length: ~{} words".format(len(text.split(" "))))
    return text

# Split text into overlapping chunks by number of tokens
def split_text_tokens(text, chunk_tokens_base, chunk_tokens_overlap, encoding, verbose=False):
    # Convert text into tokens
    tokens = encoding.encode(text)
    if verbose: print("Number of tokens: {}".format(len(tokens)))
    # Split tokens into chunks
    chunk_starts = list(range(0, len(tokens), chunk_tokens_base))
    chunk_ends = [chunk_start+chunk_tokens_base+chunk_tokens_overlap for chunk_start in chunk_starts]
    # Ensure chunk ends do not exceed text length
    chunk_ends[-1] = len(tokens)
    # Split token list into chunks
    chunks = [tokens[chunk_start:chunk_end] for chunk_start, chunk_end in zip(chunk_starts, chunk_ends)]
    # Convert chunks back into text
    chunks = [encoding.decode(chunk) for chunk in chunks]
    if verbose: print("Number of chunks: {}".format(len(chunks)))
    return(chunks)

# Split pdf text into overlapping chunks for GPT processing
def split_text(text, chunk_size_base=1000, overlap=100, verbose=False):
    # Split text and define chunk start and end indices
    words = text.split(" ")
    chunk_starts = list(range(0, len(words), chunk_size_base))
    chunk_ends = [chunk_start+chunk_size_base+overlap for chunk_start in chunk_starts]
    # Ensure chunk ends do not exceed text length
    chunk_ends[-1] = len(words)
    # Split text into chunks
    chunks = [" ".join(words[chunk_start:chunk_end]) for chunk_start, chunk_end in zip(chunk_starts, chunk_ends)]
    # Print info
    if verbose:
        print("Number of words: ~{}".format(len(words)))
        print("Number of chunks: {}".format(len(chunks)))
        print("Chunk lengths: {}".format([len(chunk) for chunk in chunks]))
        print("Chunk start indices: {}".format(chunk_starts))
        print("Chunk end indices: {}".format(chunk_ends))
    return chunks

# Generate a prompt for ChatGPT API text cleanup
def generate_prompt(text, output_format):
    prompt = "You are a research assistant tasked with accurately converting research papers from PDF to {output_format} format. You have been given the following text, which has been automatically extracted from a PDF file. Please clean up the text and convert it into {output_format} format. Make sure the result accurately and completely reflects the original text, without any errors, omissions, additions, or summarization. The text is as follows:\n\n{text}\n\nHTML:"
    return prompt.format(text=text, output_format=output_format)

# Call OpenAI API for a chat completion and return the response
def call_openai(prompt, model, max_tokens):
    response = openai.ChatCompletion.create(
        model = model,
        messages = [{"role":"user", "content":prompt}],
        temperature = 0.5,
        max_tokens = max_tokens,
    )
    output = response.choices[0].message.content.strip()
    return(output)

# Clean a single chunk of PDF text with ChatGPT API
def clean_chunk(chunk, model, encoding, max_tokens, verbose=False):
    # Generate prompt & count tokens
    prompt = generate_prompt(chunk, "text")
    prompt_tokens = encoding.encode(prompt)
    if verbose: print("Tokens in prompt: {}".format(len(prompt_tokens)))
    max_tokens_response = max_tokens - len(prompt_tokens) - 10
    # Call ChatGPT API
    cleaned_text = call_openai(prompt, model, max_tokens_response)
    cleaned_text_tokens = encoding.encode(cleaned_text)
    if verbose: print("Tokens in cleaned text: {}".format(len(cleaned_text_tokens)))
    return cleaned_text

# Clean a pre-split, multi-chunk PDF text with ChatGPT API
def clean_chunks(chunks, model, encoding, max_tokens, verbose=False):
    cleaned_chunks = []
    f = partial(clean_chunk, model=model, encoding=encoding, max_tokens=max_tokens, verbose=verbose)
    for n in range(len(chunks)):
        print("Chunk {} of {}".format(n+1, len(chunks)))
        cleaned_chunk = f(chunks[n])
        cleaned_chunks.append(cleaned_chunk)
    return cleaned_chunks
    

# Split chunk into overlapping and non-overlapping sections
def split_chunk(chunk, overlap=100, expansion_factor = 1.5, chunk_start=False,
                chunk_end=False, verbose=False):
    # Split chunk into words
    words = chunk.split(" ")
    # Determine overlap ranges
    overlap_size = overlap * expansion_factor
    break_start = overlap_size if not chunk_start else 0
    break_end = -overlap_size if not chunk_end else len(words)
    # Split chunk into overlapping and non-overlapping sections
    chunk_start = words[0:break_start]
    chunk_end = words[break_end:]
    chunk_core = words[break_start:break_end]
    # Join sub-chunks into strings
    chunk_start = " ".join(chunk_start)
    chunk_end = " ".join(chunk_end)
    chunk_core = " ".join(chunk_core)
    return [overlap_start, chunk_core, overlap_end]

# Resolve overlapping text between two chunks with ChatGPT API
def fix_chunk_overlap(chunk1, chunk2, model):
    input_text = " ".join([chunk1, chunk2])
    prompt = "You are a research assistant tasked with accurately transcribing research papers. Due to a transcription error, the following text from a research paper contains content that has been duplicated. Please clean up the text and resolve any duplications, returning the corrected result. Make sure the result accurately reflects the original text, without any errors, additions, or summarization. The text is as follows:\n\n{text}\n\nHTML:"
    prompt = prompt.format(text=input_text))
    response = call_openai(prompt, model, len(input_text.split(" "))*1.5)
    return response

# Resolve overlaps between chunks
def resolve_overlaps(chunks, model, overlap=100, expansion_factor = 1.5, verbose=False):
    # Split chunks into overlapping and non-overlapping sections
    split_chunks = []
    f = partial(split_chunk, overlap = overlap, expansion_factor = expansion_factor, 
                verbose = verbose)
    for n in range(len(chunks)):
        if n == 0: split_chunks.append(f(chunk, chunk_start=True))
        elif n == len(chunks)-1: split_chunks.append(f(chunk, chunk_end=True))
        else: split_chunks.append(f(chunk))
    # Resolve overlaps
    resolved_chunks = []
    for n in range(len(chunks)):
        if n > 0:
            chunk1 = split_chunks[n-1][2]
            chunk2 = split_chunks[n][0]
            resolved_chunk = fix_chunk_overlap(chunk1, chunk2, model)
            resolved_chunks.append(resolved_chunk)
        resolved_chunks.append(split_chunks[n][1])
    # Combine chunks into a single string
    resolved_text = " ".join(resolved_chunks)
    return resolved_text

# Save HTML to file
def save_html(html, file_path):
    with open(file_path, 'w') as file:
        file.write(html)

# Main function
def main(input_pdf, output_html, model, chunk_size_base=1000, overlap=100):
    # Convert PDF to text
    print("API key: {}".format(openai.api_key))
    print("Converting PDF to text...")
    text = pdf_to_text(input_pdf, verbose=True)
    print("...done.\n")

    # Split text into chunks
    print("Splitting text into chunks...")
    chunks = split_text(text, chunk_size_base=chunk_size_base, 
                        overlap=overlap, verbose=True)
    print("...done.\n")

    # For each chunk, clean text and convert to HTML with ChatGPT API
    print("Cleaning chunks with ChatGPT API...")
    print("Model: {}\n".format(model))
    print("...done.\n")

    # Resolve overlaps between chunks
    print("Resolving overlaps between chunks...")
    cleaned_text = resolve_overlaps(cleaned_chunks, overlap=overlap, verbose=True)
    print("...done.\n")
    print("Cleaned text length: ~{} words".format(len(text.split(" "))))

    # # Convert cleaned text to HTML
    # html = text_to_html(cleaned_text)

    # Save HTML to file
    save_html(html, output_html)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a PDF into cleaned HTML.')
    parser.add_argument('input_pdf', type=str, help='Input PDF file path')
    parser.add_argument('output_html', type=str, help='Output HTML file path')
    parser.add_argument('-m', '--model', type=str, help='OpenAI model name', default="gpt-4")
    args = parser.parse_args()
    
    main(args.input_pdf, args.output_html, args.model)
