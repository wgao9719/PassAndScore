// Copyright 2019 Google LLC & Bastiaan Konings
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "file.h"

#include "../base/log.hpp"
#include "../main.hpp"

namespace fs = boost::filesystem;

std::string GetFile(const std::string &fileName) {
  DO_VALIDATION;
  std::ifstream file;
  file.open(fileName.c_str(), std::ios::in);
  std::string str((std::istreambuf_iterator<char>(file)),
      std::istreambuf_iterator<char>());
  file.close();
  return str;
}

#include <dirent.h>
#include <sys/stat.h>

#include <dirent.h>
#include <sys/stat.h>

void GetFilesRec(std::string path, const std::string &extension,
                 std::vector<std::string> &files) {
  DO_VALIDATION;
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(path.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      if (ent->d_name[0] == '.') continue;
      std::string filename = ent->d_name;
      std::string fullPath = path + "/" + filename;
      
      struct stat st;
      if (stat(fullPath.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        GetFilesRec(fullPath, extension, files);
      } else {
        if (fullPath.length() >= extension.length() && 
            fullPath.compare(fullPath.length() - extension.length(), extension.length(), extension) == 0) {
          files.push_back(fullPath);
        }
      }
    }
    closedir(dir);
  } else {
    Log(e_Error, "DirectoryParser", "Parse",
        "Could not open directory " + path + " for reading");
  }
}

void GetFiles(std::string path, const std::string &extension,
              std::vector<std::string> &files) {
  DO_VALIDATION;
  GetFilesRec(GetGameConfig().updatePath(path), extension, files);
}
