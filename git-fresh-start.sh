#!/bin/bash
# Fresh Git Repository Script
# WARNING: This deletes ALL git history and creates a fresh repository

echo "WARNING: This will delete ALL git history!"
echo "Current files will be preserved, but all commit history will be lost."
read -p "Are you sure? (type 'yes' to continue): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Store the remote URL before deleting .git
REMOTE_URL=$(git remote get-url origin 2>/dev/null)

if [ -z "$REMOTE_URL" ]; then
    echo "No remote URL found. Please enter the remote URL:"
    read REMOTE_URL
fi

echo "Remote URL: $REMOTE_URL"

# Delete the .git directory
echo "Deleting .git directory..."
rm -rf .git

# Initialize a new git repository
echo "Initializing new git repository..."
git init

# Add the remote
echo "Adding remote origin..."
git remote add origin "$REMOTE_URL"

# Add all files (respects .gitignore)
echo "Adding files..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit - fresh start $(date '+%Y-%m-%d %H:%M:%S')"

# Push to remote (force push required since we're rewriting history)
echo "Pushing to remote..."
if timeout 60 git push -u origin master --force 2>&1; then
    echo "Successfully pushed to remote!"
    echo "Fresh repository created with clean history."
else
    echo "Warning: Failed to push to remote."
    echo "You may need to run: git push -u origin master --force"
fi
