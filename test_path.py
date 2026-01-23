
print("Start")
try:
    with open("test_write.txt", "w") as f:
        f.write("Hello from Python")
    print("Success")
except Exception as e:
    print(f"Error: {e}")
