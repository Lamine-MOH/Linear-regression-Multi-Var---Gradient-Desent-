import json
import math
import numpy as np
print()


def absolute(num):
    if(num >= 0):
        return num
    else:
        return num * (-1)


def cost(thetas, inputs, outputs, featuresNum):
    cost = np.zeros(featuresNum + 1)

    for x, y in zip(inputs, outputs):

        guess = thetas[0]
        for i in range(featuresNum):
            guess += thetas[i + 1] * x[i]

        cost[0] += (y - guess) * x[i]
        for i in range(featuresNum):
            cost[i + 1] += (y - guess) * x[i]

    return cost


def gradientDescent(inputs, outputs, stopValue=0.01):
    dataLength = len(outputs)

    featuresNum = len(inputs[0])
    thetas = np.zeros(featuresNum + 1)

    learningRate = 0.003

    while(True):
        error = cost(thetas, inputs, outputs, featuresNum)
        error[0] *= 5

        test = True
        for i in range(featuresNum + 1):
            if(absolute(error[i]) > stopValue):
                test = False
                break

        # text = ""

        if(test == True):
            break
        else:
            for i in range(featuresNum + 1):
                thetas[i] += (learningRate/dataLength) * error[i]

                # text += f"{(thetas[i])}, "

        # print(text)

    return thetas


def main():
    # get the data
    with open("data.json") as f:
        data = json.load(f)["houses"]

    featuresNames = []
    for var in data[0]["inputs"]:
        featuresNames.append(var)
    featuresNum = len(featuresNames)
    #

    # sprite the inputs and the outputs
    inputs = []
    outputs = []

    for d in data:
        case = []
        for varName in featuresNames:
            case.append(d["inputs"][varName])

        inputs.append(case)
        outputs.append(d["output"])
    #

    # call the gradient funcion
    thetas = gradientDescent(inputs, outputs, 1)

    # print the result
    print()

    text = f"H(x) = {thetas[0]}"
    for i in range(featuresNum):
        text += f" + {thetas[i + 1]} * {featuresNames[i]}"

    print(text)
    #


if __name__ == '__main__':
    main()

    print()
