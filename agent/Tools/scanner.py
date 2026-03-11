import os

def scan_project(root_folder):
   extensions = [".py",".js",".ts",".jsx",
".tsx",".java",".c",".cpp",".cc",
".h",".hpp",".cs",".go",".rs",".php",
".rb",".kt",".swift",".scala",".html",
".css",".json",".toml",".env"]
   ignore = [".venv",".git",".agent","__pycache__","venv","dist","build","node_modules"]
   collected_files = []

   for root, dirs, files in os.walk(root_folder,topdown=True):
      
      dirs[:]=[d for d in dirs if d not in ignore]

      for file in files :
         _, ext = os.path.splitext(file)
         if ext in extensions : 
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, root_folder)
            collected_files.append(relative_path)
   return collected_files

if __name__ == "__main__":
    root = os.getcwd()
    files = scan_project(root)

    print("Files discovered:")
    for f in files:
        print(f)