from sys import stdin

if __name__ == "__main__":

    # read everything from stdin until EOF
    print("READING ENTIRE STREAM")
    stream = stdin.read()
    print(stream)

    # split based on \n
    print("PRINTING LINE-BY-LINE")
    lines = stream.split("\n")
    for line in lines:
        print(line)
