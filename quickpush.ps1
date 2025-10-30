# quickpush.ps1
# A simple script to run pre-commit, commit with timestamp, and push.

# Run pre-commit hooks on all files
pre-commit run --all-files

# Add all changes to git
git add -A

# Create a timestamp for the commit message
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Commit with the timestamp in the message
git commit -m "stb $timestamp"

# Push to the current branch
git push
