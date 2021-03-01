
from tkinter import *
import cv2
import numpy as np
import sqlite3
def main():
    global scr,scr1
    try:
        scr1.destroy()
    except:
        pass
    scr=Tk()
    scr.title("Attandance System")
    scr.geometry("1920x1080")
    #screen_width = scr.winfo_screenwidth()
    #screen_height = scr.winfo_screenheight()
    from PIL import ImageTk, Image  
        
        
    l=Label(scr,text="ATTANDANCE SYSTEM",bg="blue",fg="white",font="verdana 30 bold",justify=CENTER)
    l.pack(side=TOP,fill=X)

    file = 'abh.png'

    image = Image.open(file)

    zoom = 0.2

    #multiple image size by zoom
    pixels_x, pixels_y = tuple([int(zoom * x)  for x in image.size])

    img = ImageTk.PhotoImage(image.resize((pixels_x+500, (pixels_y)))) 
    label = Label(scr, image=img)
    label.image = img
    label.pack()
    def register():
        from PIL import ImageTk, Image  
        global scr,scr1
        try:
            scr.destroy()
        except:
            pass
        scr1=Tk()
        
       
        scr1.geometry('1024x768')
        scr1.title("ATTANDANCE SYSTEM")
        global lr
        scr1.title("Register")
        scr1.geometry("1920x1080")

           
        l=Label(scr1, text="registration page",bg="red",fg="white",font="verdana 30 bold",justify=CENTER,).pack(side=TOP,fill=X)
        
            
        lr = Label(scr1, text="", font=('arial', 18))
        lr.place(x=150,y=500)
        abhi = Label(scr1, text="UserId:", font=('arial', 18), bd=18)
        abhi.place(x=200,y=200)
        e1=Entry(scr1,font=('times',16,'italic'))
        e1.place(x=350,y=220)
        abhi1 = Label(scr1, text="Name:", font=('arial', 18), bd=18)
        abhi1.place(x=200,y=250)
        e2=Entry(scr1,font=('Gender',16,'italic'))
        e2.place(x=350,y=270)
        abhi2 = Label(scr1, text="Gender:", font=('arial', 18), bd=18)
        abhi2.place(x=200,y=300)
        e3=Entry(scr1,font=('times',16,'italic'))
        e3.place(x=350,y=320)
        mobno = Label(scr1, text="Occupation:", font=('arial', 18), bd=18)
        mobno.place(x=200,y=350)
        e4=Entry(scr1,font=('times',16,'italic'))
        e4.place(x=350,y=370)
        adds = Label(scr1, text="Age:", font=('arial', 18), bd=18)
        adds.place(x=200,y=400)
        e5=Entry(scr1,font=('times',16,'italic'))
        e5.place(x=350,y=420)

            
            


        def InsertOrUpdate(Id,Name,Gender,Occupation,Age):
            import cv2
            faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            cam=cv2.VideoCapture(0)
           
            con=sqlite3.connect("FaceBase.db")
            try:
                con.execute("create table People (Id int,Name varchar(50),Gender varchar(50),Occupation varchar(50),Age int)")
            except:
                pass
            
            cmd="SELECT * FROM People WHERE ID="+str(Id)
            cursor=con.execute(cmd)
            isRecordExist=0
            for row in cursor:
                isRecordExist=1
            if(isRecordExist==1):
                cmd="UPDATE People SET Name=%r ,Gender=%r ,Occupation=%r ,Age=%r WHERE ID=%d"%(Name,Gender,Occupation,Age,int(Id))
                con.execute(cmd)
                con.commit()
                
            else:
                cmd="INSERT INTO People Values(%r,%r,%r,%r,%r)"%(Id, Name, Gender, Occupation, Age)
                con.execute(cmd)
                con.commit()
                con.close()
            sampleNum=0
           

            while(True):
                ret, img=cam.read()
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=faceDetect.detectMultiScale(gray, 1.3,5)
                for(x,y,w,h) in faces:
                    sampleNum=sampleNum+1
                    cv2.imwrite("Dataset/User."+str(Id)+"."+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.waitKey(100)
                cv2.imshow("Frame", img)
                cv2.waitKey(1)
                if(sampleNum>30):
                    break
            cam.release()
            cv2.destroyAllWindows()
            import os
            import cv2
            import numpy as np
            from PIL import Image

            recognizer=cv2.face.LBPHFaceRecognizer_create()
            path='Dataset'

            def getImagesWithID(path):
                imgPaths=[os.path.join(path,f) for f in os.listdir(path)]
                faces=[]
                Ids=[]
                for imgPath in imgPaths:
                    faceImg=Image.open(imgPath).convert("L")
                    faceNp=np.array(faceImg,'uint8')
                    ID=(int(os.path.split(imgPath)[-1].split('.')[1]))
                    faces.append(faceNp)
                    Ids.append(ID)
                    cv2.imshow("traning", faceNp)
                    cv2.waitKey(10)
                return np.array(Ids), faces

            Ids, faces=getImagesWithID(path)
            recognizer.train(faces,Ids)
            recognizer.save("recognizer/trainingData.yml")
            cv2.destroyAllWindows()
            main()

                


        #id1=input('Enter User ID')
        #name=input('Enter your Name')
        #gender=input('Enter your Gender')
        #occupation=input('Enter your Occupation')
        #Age=input('Enter your Age')
        b2=Button(scr1,text='Login',font=('arial', 12),width=14,bg='blue',fg='white',command= lambda :InsertOrUpdate(e1.get(),e2.get(),e3.get(),e4.get(),e5.get()))
        b2.place(x=420,y=450)





    def b1():
        import cv2
        import numpy as np
        import sqlite3
        from datetime import datetime

        faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cam=cv2.VideoCapture(0)
        rec=cv2.face.LBPHFaceRecognizer_create()
        rec.read("recognizer/trainingData.yml")
        def attandance(name,Rollno):
            from datetime import date

            today = date.today()
            d1 = today.strftime("%d/%m/%Y")
            with open("face.csv",'r+') as f:
                myDatalist=f.readlines()
                nameList=[]
                #date=[]
                for line in myDatalist:
                    entry=line.split(',')
                    nameList.append(entry[0])
                    #date.append(entry[0])
                if name not in nameList:
                    now=datetime.now()
                    dtString=now.strftime("%H:%M:%S")
                    f.writelines(f'\n{name},{Rollno},{d1},{dtString}')
            
            

        def getProfile(id):
            con=sqlite3.connect("FaceBase.db")
            cmd="SELECT * FROM People WHERE ID="+str(id)
            cursor=con.execute(cmd)
            profile=None
            for row in cursor:
                profile=row
            con.close()
            return profile

        id=0
        #font = cv2.cv2.InitFont(cv2.cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontcolor = (0, 255, 0)
        while(True):
            ret, img=cam.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=faceDetect.detectMultiScale(gray, 1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                id, conf=rec.predict(gray[y:y+h, x:x+w])
                profile=getProfile(id)
                if conf>60:
                    profile=None
                if (profile!=None):
                    
            
                    
                    #cv2.cv.putText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
                    cv2.putText(img, "Name:"+str(profile[1]), (x,y+h+30), fontface, 0.6, fontcolor, 2)
                    cv2.putText(img, "Age:"+str(profile[4]), (x,y+h+60), fontface, 0.6, fontcolor, 2)
                    cv2.putText(img, "Gender:"+str(profile[2]), (x,y+h+90), fontface, 0.6, fontcolor, 2)
                    cv2.putText(img, "Rollno:"+str(profile[3]), (x,y+h+120), fontface, 0.6, fontcolor, 2)
                    attandance(profile[1],profile[3])

            cv2.imshow("Frame", img)
            if(cv2.waitKey(1)==ord('q')):
                break;
        cam.release()
        cv2.destroyAllWindows()
        
       


    b1=Button(scr,text='Take Attandance',font=('arial', 15),width=17,bg='blue',fg='white',command=b1)
    b1.place(x=420,y=450)
    b2=Button(scr,text='Register',font=('arial', 15),width=17,bg='red',fg='white',command=register)
    b2.place(x=700,y=450)

    scr.mainloop()
main()
