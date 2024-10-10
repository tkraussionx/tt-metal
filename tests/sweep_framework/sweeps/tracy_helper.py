import tracy


# Function you want to profile
@tracy.profile
def profiled_function(x, y):
    return x + y


# Function that calls and profiles the other function
def calling_function():
    x, y = 5, 10
    result = profiled_function(x, y)
    print(f"Result: {result}")


if __name__ == "__main__":
    calling_function()
