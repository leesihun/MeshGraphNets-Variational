#!/bin/bash
# Add all changes
git add .

# Check if there is anything to commit
if ! git diff --cached --quiet; then
    # Commit with current date and time as the message
    git commit -m "$(date '+%Y-%m-%d %H:%M:%S')"

    # Pull remote changes first (rebase to keep linear history)
    echo "Pulling remote changes..."
    if ! git pull --rebase; then
        echo "Warning: Failed to pull from remote. You may need to manually sync."
    fi

    # Push to the current branch's upstream (with timeout)
    echo "Pushing to remote..."
    if timeout 30 git push 2>&1; then
        echo "Successfully pushed to remote!"
    else
        echo "Warning: Failed to push to remote. Changes are committed locally."
        echo "Run 'git push' manually when ready."
    fi
else
    echo "No changes to commit."
fi