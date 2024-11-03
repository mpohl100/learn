#!/bin/bash
#!/bin/bash

# Directories to search for source files
DIRECTORIES=("src" "apps")

# Function to check formatting for a given directory
check_formatting() {
    local dir=$1
    find "$dir" \( -name "*.cpp" -o -name "*.h" \) -print0 | while IFS= read -r -d '' file; do
        # Run clang-format in dry-run mode and compare
        if ! diff -u "$file" <(clang-format "$file"); then
            echo "File $file is not formatted correctly"
            exit 1
        fi
    done
}

# Loop over each directory and check formatting
for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        check_formatting "$dir"
    else
        echo "Directory $dir does not exist"
        exit 1
    fi
done

echo "All files are formatted correctly"

