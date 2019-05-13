#!/usr/bin/env python3

import epitran
import sys

def main():
  for line in sys.stdin:
    epi = epitran.Epitran('tur-Latn')
    phones = list(epi.transliterate(line))
    print (' '.join(phones))

if __name__ == '__main__':
    main()
  
