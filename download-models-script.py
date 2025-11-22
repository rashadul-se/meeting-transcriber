"""
Whisper Model Pre-download Script
==================================
Run this script ONCE before deploying to download all Whisper models.
This saves time during first-time usage and prevents crashes.

Usage:
    python download_models.py

Or download specific models:
    python download_models.py --models tiny base
"""

import whisper
import argparse
import sys

def download_model(model_name):
    """Download a specific Whisper model"""
    try:
        print(f"\n{'='*60}")
        print(f"Downloading Whisper '{model_name}' model...")
        print(f"{'='*60}")
        
        model = whisper.load_model(model_name)
        
        print(f"✅ Successfully downloaded '{model_name}' model")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"❌ Error downloading '{model_name}' model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Whisper models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tiny", "base", "small", "medium"],
        choices=["tiny", "base", "small", "medium", "large"],
        help="Models to download (default: tiny base small medium)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("   WHISPER MODEL DOWNLOADER")
    print("="*60)
    print(f"\nModels to download: {', '.join(args.models)}")
    print("\nThis may take several minutes depending on your connection...")
    
    # Model sizes
    sizes = {
        "tiny": "~39 MB",
        "base": "~74 MB",
        "small": "~244 MB",
        "medium": "~769 MB",
        "large": "~1550 MB"
    }
    
    total_size = sum([39 if m == "tiny" else 74 if m == "base" else 244 if m == "small" 
                      else 769 if m == "medium" else 1550 for m in args.models])
    
    print(f"\nEstimated total download: ~{total_size} MB\n")
    
    success_count = 0
    failed_models = []
    
    for model_name in args.models:
        print(f"\n📥 [{args.models.index(model_name) + 1}/{len(args.models)}] {model_name.upper()} ({sizes.get(model_name, 'Unknown size')})")
        
        if download_model(model_name):
            success_count += 1
        else:
            failed_models.append(model_name)
    
    # Summary
    print("\n" + "="*60)
    print("   DOWNLOAD SUMMARY")
    print("="*60)
    print(f"\n✅ Successfully downloaded: {success_count}/{len(args.models)} models")
    
    if failed_models:
        print(f"❌ Failed to download: {', '.join(failed_models)}")
        sys.exit(1)
    else:
        print("\n🎉 All models downloaded successfully!")
        print("\nYou can now run the Streamlit app:")
        print("   streamlit run app.py --server.maxUploadSize=500")
        sys.exit(0)

if __name__ == "__main__":
    main()
