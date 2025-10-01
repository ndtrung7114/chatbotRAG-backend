from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import time
import base64
import re
from groq import Groq
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # Load environment variables from .env file if present

def convert_pdf_to_md(pdf_path: str) -> str:
    """Convert PDF to MD with image summaries. Returns MD string. (Server-adapted from select_file)"""
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF not found: {pdf_path}")

    # Enable image extraction in pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_formula_enrichment = True
    pipeline_options.generate_picture_images = True  # Key: enable image extraction

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    })

    start_time = time.time()
    result = converter.convert(pdf_path)
    end_time = time.time()

    # Export to Markdown (placeholders like <!-- image --> will be present)
    md = result.document.export_to_markdown()

    # Extract images in a list of dicts
    images_list = []  # List to store dicts with image details

    for item, _ in result.document.iterate_items():
        if item.label == "picture":  # Targets figures/images
            image_data = item.image
            uri = str(image_data.uri)  # Data URI like 'data:image/png;base64,...'

            # Decode the base64 data
            match = re.match(r'data:image/(?P<type>.+);base64,(?P<data>.+)', uri)
            if match:
                img_type = match.group('type')  # e.g., 'png' or 'jpeg'
                img_bytes = base64.b64decode(match.group('data'))

                # Store in list
                images_list.append({
                    'page': item.prov[0].page_no if item.prov else 'Unknown',
                    'label': item.label,
                    'type': img_type,
                    'bytes': img_bytes,
                    'uri': uri
                })

    # Now, summarize images using VLM (Groq with Llama model)
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    prompt_template = """
    You are an expert research assistant in Artificial Intelligence. 
    Your task is to analyze and summarize a figure from a scientific paper.

    The figure may describe an overall architecture, workflow, plot, charts or experimental setup. 
    Provide a clear, detailed summary that helps a reader understand the design without seeing the image.

    When summarizing if figure is model architecture, include:
    - The main purpose of the figure (what problem it addresses).  
    - The overall structure (e.g., input/output, branches, modules, flows).  
    - The key components (e.g., encoders, decoders, adapters, loss functions).  
    - The interactions or data flow between components.  
    - Any special innovations or unique design choices.  
    if figure is charts, images or plot, analyze it.

    Format the summary inside **one section only**.  
    Do not create multiple headers like ## or ###.  
    Use bold or bullet points if needed.

    Now summarize the following figure:
    {image_caption_or_context}
    """

    image_summaries = []

    # Prepare list of base64 strings and types from images_list (assuming order matches placeholders)
    images = [(base64.b64encode(img['bytes']).decode('utf-8'), img['type']) for img in images_list]

    for img_b64, img_type in images:
        try:
            # Use correct MIME type based on extracted image type
            img_data_url = f"data:image/{img_type};base64,{img_b64}"

            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_template},
                            {"type": "image_url", "image_url": {"url": img_data_url}}
                        ]
                    }
                ],
                temperature=0.0,
                max_completion_tokens=512,
                top_p=1,
                stream=False,
            )

            summary = completion.choices[0].message.content
            image_summaries.append(summary)

        except Exception as e:
            print(f"Error processing image: {e}")
            image_summaries.append("Error summarizing image.")

    # Replace placeholders in Markdown with summaries
    # Assuming placeholders are "<!-- image -->" and appear in the same order as extracted images
    placeholder = "<!-- image -->"
    if len(image_summaries) > 0:
        # Split the Markdown by placeholder
        md_parts = md.split(placeholder)
        if len(md_parts) == len(image_summaries) + 1:
            updated_md = md_parts[0]
            for i in range(len(image_summaries)):
                # Insert summary (formatted nicely in Markdown)
                updated_md += f"\n**Image Summary:**\n{image_summaries[i]}\n" + md_parts[i + 1]
            md = updated_md
        else:
            print("Warning: Number of placeholders doesn't match number of summaries.")

    # Save paper to file md
    # Extract the file name from the full file path
    # file_name = Path(pdf_path).stem + ".pdf"  # Use stem + .pdf to match original basename logic
    # os.makedirs("../data", exist_ok=True)
    #
    # # Save the file in the 'data' folder with the extracted file name
    # output_path = f"data/{file_name}.md"
    # with open(output_path, "w", encoding="utf-8") as f:
    #     f.write(md)
    return md

if __name__ == "__main__":
    # For local testing: Replace with your good PDF path
    pdf_path = r"E:\Study\AI\PE-CLIP.pdf"  # Update this!
    md = convert_pdf_to_md(pdf_path)
    print(md[:1000])  # Print first 1000 characters of the Markdown