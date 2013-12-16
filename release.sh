#!/bin/bash

sed -i -e 's/-SNAPSHOT//' vcfplt.py 
version=`grep __version__ vcfplt.py | sed -e "s/__version__[ ]=[ ]'\(.*\)'/\1/"`
echo $version
# git commit and push
git add --all
git commit -m v$version
git push
# git tag and push
git tag -a v$version -m v$version
git push --tags
# update pypi
python setup.py register sdist upload
