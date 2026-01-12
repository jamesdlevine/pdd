import logging
import uuid
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Union

# Updated imports to match the new specification
from pdd.context_map_models import (
    ContextMap, Provenance, Input, Output,
    ApiStructure, PromptBreakdown, PreprocessorItem,
    PreprocessorSummary, PreprocessorSummaryExtra, FewShotExample,
    PreprocessorType, IncludeSyntax, SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

class ContextStore:
    """
    Manages the persistence, rotation, and retention of ContextMap data
    generated during pdd operations.
    """

    def __init__(self, output_file_path: Union[str, Path], retention_limit: int = 5):
        """
        Initialize the store for a specific output file (devunit).

        Args:
            output_file_path: The path to the generated code file (e.g., src/main.py).
                              Context files will be stored in .pdd_context/ relative to this.
            retention_limit: Maximum number of context files to keep for this devunit.
        """
        self.output_path = Path(output_file_path).resolve()
        self.retention_limit = retention_limit
        
        # Define storage directory: .pdd_context/ sibling to the output file
        self.context_dir = self.output_path.parent / ".pdd_context"
        self.basename = self.output_path.name

    def _ensure_directory(self) -> bool:
        """Creates the context directory if it doesn't exist."""
        try:
            self.context_dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as e:
            logger.warning(f"Failed to create context directory {self.context_dir}: {e}")
            return False

    def _get_existing_files(self) -> List[Path]:
        """
        Returns a sorted list of existing context files for this devunit.
        Sorts by the integer sequence number N in <basename>.context.<N>.json.
        """
        if not self.context_dir.exists():
            return []

        pattern = f"{self.basename}.context.*.json"
        files = []
        
        for p in self.context_dir.glob(pattern):
            try:
                # Extract N from filename.context.N.json
                parts = p.name.split('.')
                # Expected format: [basename parts] . context . N . json
                # We look for the second to last part
                if len(parts) >= 3 and parts[-2].isdigit():
                    seq_num = int(parts[-2])
                    files.append((seq_num, p))
            except (ValueError, IndexError):
                continue

        # Sort by sequence number
        files.sort(key=lambda x: x[0])
        return [f[1] for f in files]

    def _get_next_sequence_number(self, existing_files: List[Path]) -> int:
        """Determines the next monotonic sequence number."""
        if not existing_files:
            return 1
        
        try:
            # Get the sequence number of the last file
            last_file = existing_files[-1]
            parts = last_file.name.split('.')
            return int(parts[-2]) + 1
        except (ValueError, IndexError):
            return 1

    def _rotate_files(self, existing_files: List[Path]):
        """
        Enforces retention policy by deleting oldest files if limit is reached.
        Note: We delete *before* writing the new one, so we keep (limit - 1)
        to make room for the new file.
        """
        # If we are about to add one, we need to ensure we have space.
        # If count == limit, delete 1. If count > limit (edge case), delete diff + 1.
        
        while len(existing_files) >= self.retention_limit:
            file_to_remove = existing_files.pop(0) # Remove oldest (lowest N)
            try:
                file_to_remove.unlink()
                logger.debug(f"Rotated context file: removed {file_to_remove}")
            except OSError as e:
                logger.warning(f"Failed to delete old context file {file_to_remove}: {e}")

    def save(self, context_map: ContextMap) -> Optional[Path]:
        """
        Persists a ContextMap to disk with rotation logic.

        Args:
            context_map: The populated Pydantic model instance.

        Returns:
            Path to the written file, or None if an error occurred.
        """
        if not self._ensure_directory():
            return None

        try:
            existing_files = self._get_existing_files()
            
            # 1. Determine next filename
            next_seq = self._get_next_sequence_number(existing_files)
            filename = f"{self.basename}.context.{next_seq}.json"
            target_path = self.context_dir / filename

            # 2. Enforce retention (delete old files if necessary)
            self._rotate_files(existing_files)

            # 3. Write file using the model's built-in save method
            context_map.save(target_path)
            
            logger.info(f"Context map saved to {target_path}")
            return target_path

        except Exception as e:
            logger.warning(f"Failed to save context map for {self.basename}: {e}")
            return None

def cli_main():
    """
    CLI entry point for generating sample context maps.
    """
    parser = argparse.ArgumentParser(description="Context Map Sampler")
    parser.add_argument(
        "--example", 
        action="store_true", 
        help="Output an example context map to stdout (or --output file)"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        help="Write to file instead of stdout",
        default=None
    )

    args = parser.parse_args()

    if args.example:
        try:
            # Generate sample data using the model's static method
            sample_map = ContextMap.generate_sample()
            
            if args.output:
                sample_map.save(args.output)
                print(f"Sample context map written to {args.output}")
            else:
                # Print JSON to stdout
                print(sample_map.model_dump_json(indent=2))
                
        except Exception as e:
            logger.error(f"Error generating sample: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli_main()