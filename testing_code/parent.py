class Parent(object):

    # create a parent class method
    def show(self):
        print("Inside Parent class")


# create a child class
class Child(Parent):

    # Create a child class method
    def display(self):
        print("Inside Child class")


# Driver's code
obj = Child()
obj.display()

# Calling Parent class
obj.show()
