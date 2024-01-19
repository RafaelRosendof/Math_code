import os
import ebooklib
from ebooklib import epub
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def convert_epub_to_pdf(epub_file, pdf_file):
    # Read EPUB file
    book = epub.read_epub(epub_file)

    # Create PDF file
    pdf_base_name = os.path.splitext(os.path.basename(epub_file))[0]
    pdf_file = f"{pdf_base_name}.pdf"

    with open(pdf_file, 'wb') as f:
        # Create a PDF canvas
        pdf = canvas.Canvas(f, pagesize=letter)
        
        # Iterate over items in the EPUB book of type 'document'
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            # Check if the item is an EpubItem
            if isinstance(item, ebooklib.epub.EpubItem):
                # Extract and decode content (assuming it's text)
                text = item.content.decode('utf-8')
                
                # Add text to the PDF canvas (you may need to adjust coordinates)
                pdf.drawString(100, 800, text)
        
        # Save the PDF file
        pdf.save()

if __name__ == "__main__":
    # Path to the EPUB file
    epub_file = "/home/rafael/Downloads/Neural Networks and Numerical Analysis -- Bruno Després -- 2022 -- De Gruyter -- 9783110783124 -- 7f8d09fa6af8bd17020798c4ebe206b0 -- Anna’s Archive.epub"
    
    # Convert EPUB to PDF (passing None will use the default naming based on EPUB file)
    convert_epub_to_pdf(epub_file, None)
