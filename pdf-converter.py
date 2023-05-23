#!/usr/bin/env python

# Import libraries
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

#==============================================================================
# Define model class
#==============================================================================

class Model:
    def __init__(self, name, max_tokens, temperature):
        self.name = name
        self.max_tokens = max_tokens
        self.chunk_tokens_base = math.floor(self.max_tokens/3)
        self.chunk_tokens_overlap = math.floor(self.chunk_tokens_base/10)
        self.encoding = tiktoken.encoding_for_model(self.name)
        self.temperature = temperature
    def encode(self, text):
        return self.encoding.encode(text)
    def call(self, prompt):
        prompt_tokens = self.encode(prompt)
        max_tokens_response = self.max_tokens - len(prompt_tokens) - 10
        response = openai.ChatCompletion.create(
            model = self.name,
            messages = [{"role":"user", "content":prompt}],
            temperature = self.temperature,
            max_tokens = max_tokens_response,
        )
        output = response.choices[0].message.content.strip()
        return(output)

gpt_3_5_turbo = Model("gpt-3.5-turbo", 4096, 0.5)
gpt_4 = Model("gpt-4", 8192, 0.5)

#==============================================================================
# Define text classes
#==============================================================================

chunk_clean_prompt = "You are a research assistant tasked with accurately converting research papers from PDF to {output_format} format. You have been given the following text, which has been automatically extracted from a PDF file. Please clean up the text and convert it into {output_format} format. Make sure the result accurately and completely reflects the original text, without any errors, omissions, additions, or summarization. The text is as follows:\n\n{text}\n\nHTML:"

# Define Chunk class for list of tokens
class Chunk:
    def __init__(self, tokens, model, verbose):
        self.tokens = tokens
        self.model = model
        self.text = self.model.encoding.decode(self.tokens)
        self.clean_prompt = chunk_clean_prompt.format(text=self.text, output_format="text")
        self.verbose = verbose
        self.cleaned = False
    def clean(self, force=False):
        if self.verbose: print("Tokens in prompt: {}".format(len(self.model.encode(self.clean_prompt))))
        if force or not self.cleaned:
            self.clean_text = self.model.call(self.clean_prompt)
            self.clean_tokens = self.model.encoding.encode(self.clean_text)
            self.cleaned = True
        elif self.verbose: print("Chunk already cleaned.")
        if self.verbose: print("Tokens in clean text: {}".format(len(self.clean_tokens)))

class PdfText:
    def __init__(self, input_path, verbose):
        self.verbose = verbose
        self.input_path = input_path
        self.pdf = pypdf.PdfReader(self.input_path)
        if self.verbose: print("Number of pages: {}".format(len(self.pdf.pages)))
        self.text_raw = "".join([page.extract_text()+"\n" for page in self.pdf.pages])
    def set_model(self, model):
        self.model = model
        self.tokens_raw = model.encode(self.text_raw)
        if self.verbose: print("Number of tokens: {}".format(len(self.tokens_raw)))
    def split_raw(self):
        # Define encoding
        tokens = self.tokens_raw
        # Define chunk indices
        base = self.model.chunk_tokens_base
        skip = base + self.model.chunk_tokens_overlap
        chunk_starts = list(range(0, len(tokens), base))
        chunk_ends = [chunk_start+skip for chunk_start in chunk_starts]
        chunk_ends[-1] = len(tokens)
        # Split token list into chunks
        chunk_tokens = [tokens[chunk_start:chunk_end] for chunk_start, chunk_end in zip(chunk_starts, chunk_ends)]
        self.chunks = [Chunk(chunk_token, self.model, self.verbose) for chunk_token in chunk_tokens]
        if self.verbose: print("Number of chunks: {}".format(len(self.chunks)))
    def clean(self, n_attempts = 10, force=False):
        for attempt in range(n_attempts):
            status = [chunk.cleaned for chunk in self.chunks]
            if all(status): break
            for chunk in self.chunks:
                if self.verbose: print("Cleaning chunk {}/{}".format(self.chunks.index(chunk)+1, len(self.chunks)))
                try:
                    chunk.clean(force=force)
                except:
                    print("Chunk {} failed to clean.".format(self.chunks.index(chunk)+1))
                    continue
        if self.verbose:
            status = [chunk.cleaned for chunk in self.chunks]
            if all(status): print("All chunks cleaned successfully.")
            else: print("Some chunks failed to clean.")


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
