import sys
import subprocess

def bump_version(bump_type):
    if bump_type not in ['patch', 'minor', 'major']:
        print("Usage: python bump_version.py [patch|minor|major]")
        sys.exit(1)
    
    # Ensure bump2version is installed
    try:
        subprocess.run(['bump2version', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("bump2version not found. Installing...")
        subprocess.run(['pip', 'install', '--upgrade', 'bump2version'], check=True)
    
    # Bump the version
    subprocess.run(['bump2version', bump_type], check=True)
    
    # Push the changes and tags to GitHub
    subprocess.run(['git', 'push', 'origin', 'main', '--tags'], check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py [patch|minor|major]")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    bump_version(bump_type)
