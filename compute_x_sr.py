dict = {0: [0, 54.6968], 1: [0, 142.228], 2: [0, 135.438], 3: [0, 138.145], 4: [0, 90.8179], 5: [0, 153.027], 6: [0, 134.671], 7: [0, 94.8257], 8: [0, 166.258], 9: [0, 142.429]}

x = []
for i in dict.keys():
    x.append(dict[i][1])

x_sr = sum(x)/len(x)
div = 0
for i in range(len(x)):
    div += (x[i]-x_sr)**2
div= div**0.5
print(x_sr,div)
