# FashQuick

An AI-powered fashion virtual try-on application that uses SAM2 for clothing segmentation and Stable Diffusion for generating fashion models.

## Features

- Automatic clothing segmentation using SAM2
- High-quality fashion model generation with Stable Diffusion and ControlNet
- Advanced mask refinement and feature extraction
- Memory-efficient processing for large images
- Easy-to-use CLI interface

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 5GB disk space for models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fashquick.git
cd fashquick
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
python -m app.cli download-models
```

## Usage

### Web Interface

Start the web server:
```bash
python -m app.web
```

Then open your browser to `http://localhost:8081` to access the web interface. You can:
1. Upload a clothing image
2. Add an optional custom prompt
3. Click 'Generate' to create a fashion model wearing the clothing


### Generate Fashion Model

Generate a fashion model wearing the input clothing:

```bash
python -m app.cli generate path/to/clothing.jpg
```

Options:
- `-o, --output`: Specify output image path
- `-p, --prompt`: Custom generation prompt
- `-n, --num-images`: Number of images to generate (default: 1)
- `--device`: Device to use (cuda/cpu)
- `--debug`: Enable debug mode

### Segment Clothing

Extract clothing mask from an image:

```bash
python -m app.cli segment path/to/clothing.jpg
```

Options:
- `-o, --output`: Specify output mask path
- `--device`: Device to use (cuda/cpu)
- `--debug`: Enable debug mode

## Architecture

The application is organized into several modules:

- `app/`: Main application code and CLI interface
- `modules/`: Core processing modules
  - `preprocessing/`: Image preprocessing utilities
  - `segmentation/`: SAM2 segmentation and mask refinement
  - `feature_extraction/`: Color, pattern, and texture analysis
  - `generation/`: Stable Diffusion and ControlNet handling
- `utils/`: Utility modules
  - Model management
  - Memory optimization
  - Input validation

## Examples

1. Basic usage:
```bash
python -m app.cli generate clothing.jpg
```

2. Generate multiple variations:
```bash
python -m app.cli generate clothing.jpg -n 3
```

3. Use custom prompt:
```bash
python -m app.cli generate clothing.jpg -p "professional fashion model in studio lighting"
```

4. Extract clothing mask:
```bash
python -m app.cli segment clothing.jpg -o mask.png
```

## How It Works

1. **Preprocessing**: The input clothing image is loaded and preprocessed for model input.

2. **Segmentation**: SAM2 segments the clothing from the image, and the mask is refined using advanced techniques.

3. **Feature Extraction**: The system analyzes:
   - Color palette and distribution
   - Pattern type and complexity
   - Texture characteristics

4. **Generation**: Using the extracted features and mask:
   - ControlNet guides the generation process
   - Stable Diffusion creates the fashion model
   - Post-processing enhances the final result

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Segment Anything Model (SAM)](https://segment-anything.com/)
- [Stable Diffusion](https://stability.ai/)
- [ControlNet](https://github.com/lllyasviel/ControlNet)