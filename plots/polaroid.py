from PIL import Image, ImageDraw, ImageFont

def create_polaroid(input_path, output_path, text="" , font_size = 500):
    # Open the original image
    img = Image.open(input_path)
    
    # Define the size of the Polaroid border
    border_size = 30
    bottom_border_extra = 50
    
    # Calculate the size of the new image with borders
    polaroid_width = img.width + 2 * border_size
    polaroid_height = img.height + 2 * border_size + bottom_border_extra
    
    # Create a new image with white background
    polaroid = Image.new('RGB', (polaroid_width, polaroid_height), color='white')
    
    # Paste the original image onto the new image
    polaroid.paste(img, (border_size, border_size))
    
    # Draw the text (if any) on the bottom border
    if text:
        draw = ImageDraw.Draw(polaroid)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_x = (polaroid_width - text_width) // 2
        text_y = img.height + border_size + (bottom_border_extra - text_height) // 2
        draw.text((text_x, text_y), text, fill="black", font=font)
    
    # Save the final image
    polaroid.save(output_path)

# Example usage
#create_polaroid("/home/rafael/Downloads/teste.jpeg", "polaroid.jpg", text="My Polaroid")

create_polaroid("/home/rafael/Downloads/nossa.jpg", "nosso_polaroid.jpg", text="Meu Amor",font_size = 500)
