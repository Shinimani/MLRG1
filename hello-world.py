x=3
print(type(x))
print(x,x+1,x*2,x**2)
hello = 'hello'    # String literals can use single quotes
world = "world"    # or double quotes; it does not matter.
print()
print(hello)       # Prints "hello"
print(len(hello))  # String length; prints "5"
hw = hello + ' ' + world  # String concatenation
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"