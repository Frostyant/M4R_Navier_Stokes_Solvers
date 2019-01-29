class AClass():
    b = 20
    def __init__(self):
        AClass.b += 1
        print(AClass.b)

test = AClass()
test = AClass()
test = AClass()
test = AClass()
