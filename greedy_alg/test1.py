# Python program to demonstrate
# use of class method and static method.
from datetime import date
import argparse

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    # a class method to create a Person object by birth year.
    @classmethod
    def fromBirthYear(cls, name, year):
        return cls(name, date.today().year - year)
    
    # a static method to check if a Person is adult or not.
    @staticmethod
    def isAdult(age):
        return age > 18

    def show(self):
        print(self.isAdult(self.age))

person1 = Person('mayank', 21)
person2 = Person.fromBirthYear('mayank', 1996)

print (person1.age)
print (person2.age)

# print the result
print (Person.isAdult(22))
person1.show()

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--feature', dest='feature', action='store_true')
parser.add_argument('--no-feature', dest='feature', action='store_false')
parser.set_defaults(feature=True)
args = parser.parse_args()
print(args.feature)
