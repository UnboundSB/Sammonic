#!/usr/bin/env python3
while True:
    user_input = input("Enter text (Ctrl+C to stop): ")
    with open("script.txt", "a", encoding="utf-8") as f:
        f.write(user_input + "\n")