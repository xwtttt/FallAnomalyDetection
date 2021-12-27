from math import pow
def calc_ADXL345(Range,Resolution, AD):
    return ((2*Range)/(pow(2, Resolution)))*AD



print(calc_ADXL345(8, 14, -34))
print(calc_ADXL345(16, 13,  -10))
# print(calc_MMA8451Q(76))