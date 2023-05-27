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
gpt_4_small = Model("gpt-4", 4096, 0.5)
gpt_3_5_turbo_small = Model("gpt-3.5-turbo", 2048, 0.5)

#==============================================================================
# Define text classes
#==============================================================================

chunk_clean_prompt = "You are a research assistant tasked with accurately converting research papers from PDF to {output_format} format. You have been given the following text, which has been automatically extracted from a PDF file. Please clean up the text and convert it into {output_format} format. Make sure the result accurately and completely reflects the original text, without any errors, omissions, additions, or summarization. The text is as follows:\n\n{text}\n\nCleaned text:"
overlap_resolve_prompt = "You are a research assistant tasked with accurately transcribing research papers. Due to a transcription error, the following text from a research paper contains content that has been duplicated. Please clean up the text and resolve any duplications, returning the corrected result. Make sure the result otherwise accurately reflects the original text, without any errors, additions, or summarization. Do not modify the start or end of the text in any way. The text is as follows:\n\n{text}\n\nCorrected text:"

# Define Chunk class for list of tokens
class Chunk:
    def __init__(self, tokens, model, first, last, verbose):
        self.tokens = tokens # List of tokens defining chunk
        self.model = model # Model used to process chunk (defines encoding)
        self.text = self.model.encoding.decode(self.tokens) # Text of chunk
        self.clean_prompt = chunk_clean_prompt.format(text=self.text, output_format="text") # Prompt used for cleaning command
        self.verbose = verbose # Verbose output?
        self.first = first # Is this the first chunk?
        self.last = last # Is this the last chunk?
        self.cleaned = False # Has the chunk been cleaned?
    def clean(self, force=False):
        if self.verbose: print("Tokens in prompt: {}".format(len(self.model.encode(self.clean_prompt))))
        if force or not self.cleaned:
            self.clean_text = self.model.call(self.clean_prompt)
            self.clean_tokens = self.model.encoding.encode(self.clean_text)
            self.cleaned = True
        elif self.verbose: print("Chunk already cleaned.")
        if self.verbose: print("Tokens in clean text: {}".format(len(self.clean_tokens)))
    def split(self, expansion=1.5):
        # Split cleaned chunk into overlapping and non-overlapping sections
        if not self.cleaned: raise Exception("Chunk must be cleaned before splitting.")
        # Determine overlap ranges
        overlap_size = math.floor(self.model.chunk_tokens_overlap * expansion)
        break_start = overlap_size if not self.first else 0
        break_end = -overlap_size if not self.last else len(self.clean_tokens)
        # Split chunk
        chunk_prefix = self.clean_tokens[0:break_start]
        chunk_core = self.clean_tokens[break_start:break_end]
        chunk_suffix = self.clean_tokens[break_end:]
        return {"prefix":chunk_prefix, "core":chunk_core, "suffix":chunk_suffix}
    
class Overlap:
    def __init__(self, chunk1, chunk2, verbose):
        assert chunk1.model == chunk2.model
        self.model = chunk1.model
        suffix = chunk1.split()["suffix"]
        prefix = chunk2.split()["prefix"]
        self.tokens = suffix + prefix
        self.text = self.model.encoding.decode(self.tokens)
        self.resolve_prompt = overlap_resolve_prompt.format(text=self.text)
        self.verbose = verbose
        self.resolved = False
    def resolve(self, force=False):
        if self.verbose: print("Tokens in prompt: {}".format(len(self.model.encode(self.resolve_prompt))))
        if force or not self.resolved:
            self.resolved_text = self.model.call(self.resolve_prompt)
            self.resolved_tokens = self.model.encoding.encode(self.resolved_text)
            self.resolved = True
        elif self.verbose: print("Overlap already resolved.")
        if self.verbose: print("Tokens in resolved text: {}".format(len(self.resolved_tokens)))

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
        chunks = []
        for n in range(len(chunk_tokens)):
            first = True if n == 0 else False
            last = True if n == len(chunk_tokens)-1 else False
            chunk = Chunk(chunk_tokens[n], self.model, first, last, self.verbose)
            chunks.append(chunk)
        self.chunks = chunks
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
    def resolve(self, n_attempts = 10, force=False):
        # Define cores and overlaps
        split_chunks = [c.split() for c in self.chunks]
        cores = [c["core"] for c in split_chunks]
        overlaps = [Overlap(self.chunks[n], self.chunks[n+1], self.verbose) for n in range(len(self.chunks)-1)]
        assert len(cores) == (len(overlaps)+1)
        # Resolve overlaps with ChatGPT
        for attempt in range(n_attempts):
            status = [overlap.resolved for overlap in overlaps]
            if all(status): break
            for overlap in overlaps:
                if self.verbose: print("Resolving overlap {}/{}".format(overlaps.index(overlap)+1, len(overlaps)))
                try:
                    overlap.resolve(force=force)
                except:
                    print("Overlap {} failed to resolve.".format(overlaps.index(overlap)+1))
                    continue
        status = [overlap.resolved for overlap in overlaps]
        if not all(status): raise Exception("Some overlaps failed to resolve.")
        elif self.verbose: print("All overlaps resolved successfully.")
        self.resolved_cores = cores
        self.resolved_overlaps = overlaps
    def reconstruct(self):
        assert len(self.resolved_cores) == len(self.resolved_overlaps)+1
        cores = self.resolved_cores
        overlaps = self.resolved_overlaps
        space_token = self.model.encoding.encode(" ")
        output_tokens = []
        for n in range(len(overlaps)):
            output_tokens += cores[n] + space_token
            output_tokens += overlaps[n].resolved_tokens + space_token
        output_tokens += cores[-1]
        self.reconstructed_tokens = output_tokens
        self.reconstructed_text = self.model.encoding.decode(self.reconstructed_tokens)
        if self.verbose: print("Tokens in reconstructed text: {}".format(len(self.reconstructed_tokens)))

# Main function
def main(input_pdf, output_text, model, max_attempts=10, verbose=True):
    # Import PDF file and define model
    print("Importing and pre-processing PDF file...")
    text = PdfText(input_pdf, verbose=verbose)
    text.set_model(model)
    text.split_raw()
    print("...done.\n")
    # Clean text
    print("Cleaning text...")
    text.clean(n_attempts=max_attempts, force=False)
    print("...done.\n")
    # Resolve overlaps
    print("Resolving overlaps...")
    text.resolve(n_attempts=max_attempts, force=False)
    print("...done.\n")
    # Reconstruct text
    print("Reconstructing text...")
    text.reconstruct()
    print("...done.\n")
    # Export text
    print("Exporting text...")
    with open(output_text, "w") as f:
        f.write(text.reconstructed_text)
    print("...done.\n")
    return(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a PDF into cleaned HTML.')
    parser.add_argument('input_pdf', type=str, help='Input PDF file path')
    parser.add_argument('output_html', type=str, help='Output HTML file path')
    parser.add_argument('-4', '--gpt4', type=bool, help='Use GPT-4 model (default: gpt-3.5-turbo)', default=False)
    parser.add_argument('-m', '--max_attempts', type=int, help='Maximum number of attempts to clean and resolve text (default: 20)', default=20)
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output (default: False)', default=False)
    args = parser.parse_args()
    # Define model
    model = gpt_4_small if args.gpt4 else gpt_3_5_turbo
    # Run main function
    main(args.input_pdf, args.output_html, model, args.max_attempts, args.verbose)
