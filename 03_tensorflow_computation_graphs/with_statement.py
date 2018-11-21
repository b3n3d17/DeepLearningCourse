class A():

    def __init__(self, param1, param2):
        print("init()")
        self.param1 = param1
        self.param2 = param2


    def __enter__(self):
        print("enter()")
        return self


    def __exit__(self, *args):
        print("exit()")


    def foo(self):
        print("foo()")



def main():
    obj = A(10,20)
    print(">before with-statement")
    with obj:
        print(">within with-statement")
        obj.foo()
        f = open('foo2.txt','r')
        print(">at end of with-statement")
    print(">after with-statement")

main()