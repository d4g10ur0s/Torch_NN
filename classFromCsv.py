import os
import pandas as pd

def createSetter(attrName, file):
    file.write("\tdef set_"+str(attrName)+"("+attrName+"):\n")
    file.write("\t\tself."+str(attrName)+"="+attrName+"\n\n")

def createGetter(attrName, file):
    file.write("\tdef get_"+str(attrName)+"("+attrName+"):\n")
    file.write("\t\treturn self."+str(attrName)+"\n\n")

def main():
    while 1 :
        dataPath = input('Give path for data : ')
        print("Opening : " + str(dataPath))
        try :
            data = pd.read_csv(dataPath)
            # read data keys to create class attributes
            classAttr = []
            for i in data.keys():
                # replace #,- and space with underscore '_'
                newChar = str(i).replace('#', '_')
                newChar = str(newChar).replace(' ', '_')
                newChar = str(newChar).replace('-', '_')
                # replace words that are python keywords with other format
                if newChar=="class" :
                    newChar = "class_type"
                    print(newChar)
                classAttr.append(newChar)
            # make the parameter string for init
            paramString = ""
            for i in range(len(classAttr)):
                if i < len(classAttr)-1:
                    paramString = paramString + str(classAttr[i]) +" ,"
                else :
                    paramString = paramString + str(classAttr[i])
            print("Parameter String : " + paramString)
            # give the class name
            className = input("Give class name : ")
            # create class file
            f = open(className + ".py" , "w+")
            # write class definition
            f.write("class " + className + "():\n\n")
            f.write("\t# constructor of " + className+'\n')
            f.write("\tdef __init__(" + paramString +"):\n")
            for k in classAttr :
                f.write("\t\tself."+str(k)+"="+str(k)+"\n")
            f.write("\n")
            # create setter and getter
            if input("Create setter and getter : ") == 'y':
                for i in range(len(classAttr)):
                    createSetter(classAttr[i],f)
                    createGetter(classAttr[i],f)
            f.close()
        except ValueError :
            print("Error occured")
        #print('Wrong path for data')

if __name__=="__main__":
    main()
