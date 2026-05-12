# Demo script showing various ways to run the enhanced pipeline

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display results"""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Command completed successfully")
            # Print last few lines of output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-10:]:  # Show last 10 lines
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ Command failed")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Error running command: {e}")

def main():
    """Demonstrate various pipeline options"""
    
    python_path = sys.executable
    script_path = "Code/language/data_loader.py"
    
    print("🚀 Enhanced RAG Pipeline - Command Line Demo")
    print("=" * 70)
    
    # Show help
    run_command(
        f"{python_path} {script_path} --help",
        "Show help and available options"
    )
    
    # Load existing pipeline (fastest option)
    run_command(
        f"{python_path} {script_path} --load-existing --log-level INFO",
        "Load existing pipeline with INFO logging"
    )
    
    # Run with skip existing and no tests (for quick validation)
    run_command(
        f"{python_path} {script_path} --skip-existing --no-tests --log-level INFO",
        "Quick run: skip existing documents and tests"
    )
    
    # Run with custom queries
    custom_queries = [
        "What defects disqualify mangoes for export?",
        "What temperature should be maintained during mango transport?"
    ]
    
    query_args = " ".join([f'"{q}"' for q in custom_queries])
    run_command(
        f"{python_path} {script_path} --load-existing --custom-queries {query_args}",
        "Test with custom queries"
    )
    
    print("\n" + "=" * 70)
    print("🎉 Demo completed! Your pipeline now supports:")
    print("   ✅ Command line arguments")
    print("   ✅ Proper logging with timestamps")
    print("   ✅ Performance timing and statistics")
    print("   ✅ Skip existing processing for faster runs")
    print("   ✅ Custom query testing")
    print("   ✅ Better error handling and validation")
    print("   ✅ Progress tracking")

if __name__ == "__main__":
    main()