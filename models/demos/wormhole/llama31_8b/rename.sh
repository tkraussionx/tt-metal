#!/bin/bash

# Find all files and directories recursively
find . -depth -name "*llama*" | while IFS= read -r file; do
    # Generate new filename by replacing 'mistral' with 'llama31_8b'
    new_file=$(echo "$file" | sed 's/llama31_8b/llama/g')
    # Rename the file or directory
    mv "$file" "$new_file"
done
