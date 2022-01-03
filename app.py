from flask import Flask, render_template,session, redirect, url_for, escape, request
import rs as model
app = Flask(__name__)
pop=[]
user=[]
item=[]
mol=[]
cont=[]
@app.route('/',methods=['GET','POST'])
def index():
    global pop
    global user
    global item
    global mol
    global cont
    if request.method == 'POST':
        print("inside index")
        un=request.form.get('username')
        pop,user,item,mol,cont=(model.main(un))
        # task = [elem for twod in task for elem in twod]
        print('popularity---------')
        print(pop)
        print("user ------------")
        print(user)
        print("item--------------")
        print(item)
        print("mol--------")
        print(mol)
    return render_template('index.html',task=pop,task1=user,task2=item,task3=mol,task4=cont)
    





if __name__ == '__main__':
    app.run(debug=True)