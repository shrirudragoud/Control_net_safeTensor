import click
import logging
from pathlib import Path
from PIL import Image
import sys
from typing import Optional
from .config import Config
from .main import FashionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """FashQuick - AI Fashion Virtual Try-On Tool"""
    pass

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output image path')
@click.option('--prompt', '-p', type=str, help='Custom generation prompt')
@click.option('--num-images', '-n', type=int, default=1, help='Number of images to generate')
@click.option('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def generate(
    image_path: str,
    output: Optional[str] = None,
    prompt: Optional[str] = None,
    num_images: int = 1,
    device: str = 'cuda',
    debug: bool = False
):
    """Generate fashion model images from clothing input"""
    try:
        # Set debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            
        # Load configuration
        config = Config()
        
        # Update device configuration
        config.update_config({
            'sam_config': {'device': device},
            'sd_config': {'device': device}
        })
        
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = FashionPipeline(config.get_config())
        
        # Process image
        logger.info("Processing input image...")
        results = pipeline.process_image(
            image_path=image_path,
            prompt=prompt,
            num_images=num_images
        )
        
        # Save results
        if output:
            output_path = Path(output)
        else:
            # Create default output path
            input_path = Path(image_path)
            output_dir = input_path.parent / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            if num_images == 1:
                output_path = output_dir / f"{input_path.stem}_result.png"
            else:
                # Save multiple images with index
                for idx, img in enumerate(results):
                    out_path = output_dir / f"{input_path.stem}_result_{idx+1}.png"
                    logger.info(f"Saving result to {out_path}")
                    img.save(out_path)
                return
        
        # Save single image
        if num_images == 1:
            logger.info(f"Saving result to {output_path}")
            results[0].save(output_path)
            
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output mask path')
@click.option('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def segment(
    image_path: str,
    output: Optional[str] = None,
    device: str = 'cuda',
    debug: bool = False
):
    """Generate clothing segmentation mask"""
    try:
        # Set debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            
        # Load configuration
        config = Config()
        config.update_config({
            'sam_config': {'device': device}
        })
        
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = FashionPipeline(config.get_config())
        
        # Load and preprocess image
        logger.info("Processing input image...")
        image = Image.open(image_path)
        masks = pipeline._segment_clothing(image)
        refined_mask = pipeline._refine_mask(masks)
        
        # Create output path if not provided
        if output:
            output_path = Path(output)
        else:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_mask.png"
        
        # Save mask
        logger.info(f"Saving mask to {output_path}")
        mask_image = Image.fromarray((refined_mask * 255).astype('uint8'))
        mask_image.save(output_path)
        
    except Exception as e:
        logger.error(f"Error during segmentation: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
def download_models(device: str = 'cuda'):
    """Download required AI models"""
    try:
        # Load configuration
        config = Config()
        config.update_config({
            'sam_config': {'device': device},
            'sd_config': {'device': device}
        })
        
        # Initialize pipeline to trigger model downloads
        logger.info("Downloading models...")
        pipeline = FashionPipeline(config.get_config())
        logger.info("Models downloaded successfully")
        
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        sys.exit(1)

@cli.command()
def version():
    """Show version information"""
    click.echo("FashQuick v1.0.0")

def main():
    cli()

if __name__ == '__main__':
    main()