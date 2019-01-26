class AClass():
    def __init__(self):
        self.a = 2
        b = self.a*2

    def mod(self):
        self.a = 2*self.a

test = AClass()

test.mod()
