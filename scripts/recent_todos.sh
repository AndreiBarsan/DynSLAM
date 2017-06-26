#!/usr/bin/env bash
# Finds recently changed TODOs. Not perfect, but close nuff.

git log --patch --color=always HEAD@{8}..HEAD | less +/TODO
