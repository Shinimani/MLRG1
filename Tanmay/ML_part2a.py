import sys
import numpy as np
import csv

trainfile = sys.argv[1]
testfile = sys.argv[2]
param = sys.argv[3]
outputfile = sys.argv[4]
weightfile = sys.argv[5]

#SOFTMAX DEFINED
def softmax(X):
    y = np.atleast_2d(X)
    axis = 1
    # subtract the max for numerical stability
    #y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = 1), axis)
    #ax_sum = np.sum(y, axis = axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    #if len(X.shape) == 1: p = p.flatten()
    return p

                    #x = np.array([[1004,1005],[3,4]])
                    #print(softmax(x))

#CSV TO NUMPY ARRAY (TRAIN)
train_file = open(trainfile, "r")
create = False
ll = 0

l00 = 0
l11 = 0
l22 = 0
l33 = 0
l44 = 0

for line in train_file:
    par_list = line.rstrip().split(',')
    temp_lab = np.array(np.zeros((1, 5)))
    temp_par = np.array(np.zeros((1, 27)))
    #if len(par_list) != 9 and line.strip() != '':
    if len(par_list) == 9:

        if par_list[0] == 'usual':
            temp_par[0][0] = 1
        elif par_list[0] == 'pretentious':
            temp_par[0][1] = 1
        elif par_list[0] == 'great_pret':
            temp_par[0][2] = 1

        if par_list[1] == 'proper':
            temp_par[0][3] = 1
        elif par_list[1] == 'less_proper':
            temp_par[0][4] = 1
        elif par_list[1] == 'improper':
            temp_par[0][5] = 1
        elif par_list[1] == 'critical':
            temp_par[0][6] = 1
        elif par_list[1] == 'very_crit':
            temp_par[0][7] = 1

        if par_list[2] == 'complete':
            temp_par[0][8] = 1
        elif par_list[2] == 'completed':
            temp_par[0][9] = 1
        elif par_list[2] == 'incomplete':
            temp_par[0][10] = 1
        elif par_list[2] == 'foster':
            temp_par[0][11] = 1

        if par_list[3] == '1':
            temp_par[0][12] = 1
            #print('True')
        elif par_list[3] == '2':
            temp_par[0][13] = 1
            #print('False')
        elif par_list[3] == '3':
            temp_par[0][14] = 1
        elif par_list[3] == 'more':
            temp_par[0][15] = 1

        if par_list[4] == 'convenient':
            temp_par[0][16] = 1
        elif par_list[4] == 'less_conv':
            temp_par[0][17] = 1
        elif par_list[4] == 'critical':
            temp_par[0][18] = 1

        if par_list[5] == 'convenient':
            temp_par[0][19] = 1
        elif par_list[5] == 'inconv':
            temp_par[0][20] = 1

        if par_list[6] == 'nonprob':
            temp_par[0][21] = 1
            #print(True)
        elif par_list[6] == 'slightly_prob':
            temp_par[0][22] = 1
        elif par_list[6] == 'problematic':
            temp_par[0][23] = 1

        if par_list[7] == 'recommended':
            temp_par[0][24] = 1
        elif par_list[7] == 'priority':
            temp_par[0][25] = 1
        elif par_list[7] == 'not_recom':
            temp_par[0][26] = 1

        if par_list[8] == 'not_recom':
            temp_lab[0][0] = 1
            l00 += 1
        elif par_list[8] == 'recommend':
            temp_lab[0][1] = 1
            l11 += 1
        elif par_list[8] == 'very_recom':
            temp_lab[0][2] = 1
            l22 += 1
        elif par_list[8] == 'priority':
            temp_lab[0][3] = 1
            l33 += 1
        elif par_list[8] == 'spec_prior':
            temp_lab[0][4] = 1
            l44 += 1

        if not create:
            par = np.array(np.zeros((1,27)))
            lab = np.array(np.zeros((1,5)))
            #print(par.shape)
            par[0,:] = temp_par
            lab[0,:] = temp_lab
            #print(temp_par)
            #print(temp_lab)
            create = True;
        else:
            #print(temp_par)
            #print(temp_lab)
            par = np.r_[par,temp_par]
            lab = np.r_[lab,temp_lab]
        ll += 1

#print(l00,' -> ',l11,' -> ',l22,' -> ',l33,' -> ',l44)
#print(l00 + l11 + l22 + l33 + l44)
par = np.c_[np.ones(par.shape[0]),par]
weight = np.zeros((5,28))

#CSV TO NUMPY ARRAY (TEST)
test_file = open(testfile, "r")
create = False
ll = 0

for linee in test_file:
    par_list = linee.rstrip().split(',')
    temp_par = np.array(np.zeros((1, 27)))
    #if len(par_list) != 9 and line.strip() != '':
    if len(par_list) == 8:

        if par_list[0] == 'usual':
            temp_par[0][0] = 1
        elif par_list[0] == 'pretentious':
            temp_par[0][1] = 1
        elif par_list[0] == 'great_pret':
            temp_par[0][2] = 1

        if par_list[1] == 'proper':
            temp_par[0][3] = 1
        elif par_list[1] == 'less_proper':
            temp_par[0][4] = 1
        elif par_list[1] == 'improper':
            temp_par[0][5] = 1
        elif par_list[1] == 'critical':
            temp_par[0][6] = 1
        elif par_list[1] == 'very_crit':
            temp_par[0][7] = 1

        if par_list[2] == 'complete':
            temp_par[0][8] = 1
        elif par_list[2] == 'completed':
            temp_par[0][9] = 1
        elif par_list[2] == 'incomplete':
            temp_par[0][10] = 1
        elif par_list[2] == 'foster':
            temp_par[0][11] = 1

        if par_list[3] == '1':
            temp_par[0][12] = 1
            #print('True')
        elif par_list[3] == '2':
            temp_par[0][13] = 1
            #print('False')
        elif par_list[3] == '3':
            temp_par[0][14] = 1
        elif par_list[3] == 'more':
            temp_par[0][15] = 1

        if par_list[4] == 'convenient':
            temp_par[0][16] = 1
        elif par_list[4] == 'less_conv':
            temp_par[0][17] = 1
        elif par_list[4] == 'critical':
            temp_par[0][18] = 1

        if par_list[5] == 'convenient':
            temp_par[0][19] = 1
        elif par_list[5] == 'inconv':
            temp_par[0][20] = 1

        if par_list[6] == 'nonprob':
            temp_par[0][21] = 1
        elif par_list[6] == 'slightly_prob':
            temp_par[0][22] = 1
        elif par_list[6] == 'problematic':
            temp_par[0][23] = 1

        if par_list[7] == 'recommended':
            temp_par[0][24] = 1
        elif par_list[7] == 'priority':
            temp_par[0][25] = 1
        elif par_list[7] == 'not_recom':
            temp_par[0][26] = 1

        if not create:
            #print('df\n')
            test_X = np.array(np.zeros((1,27)))
            #print(par.shape)
            test_X[0,:] = temp_par
            #print(temp_par)
            create = True;
        else:
            #print(temp_par)
            test_X = np.r_[test_X,temp_par]
        ll += 1

test_X = np.c_[np.ones(test_X.shape[0]),test_X]
print(test_X.shape)
param_file = open(param, "r")
l_no = 0
for line in param_file:
    if l_no == 0:
        lrs = line.rstrip()
        #print('yes')
        l_no = 1
        continue

    if l_no == 1:
        if lrs == '1' or lrs == '2':
            lr = float(line.rstrip())
            l_no = 2
        else:
            #print(lrs,'f')
            alpha_beta = line.rstrip().split(',')
            #print(alpha_beta)
            lr = float(alpha_beta[0])
            alpha = float(alpha_beta[1])
            beta = float(alpha_beta[2])
            l_no = 2
        continue

    if l_no == 2:
        max_iter = int(line.rstrip())

#print('Done')
#print(lrs, ' -> s',lr,' -> ', max_iter)

a = np.dot(par,np.transpose(weight))
#'''
if lrs == '1':
    #lr = 2
    for iterr in range(max_iter):
        a = np.dot(par,np.transpose(weight))
        error = np.subtract(softmax(a), lab)
        gradient = np.dot(np.transpose(error),par)/(par.shape[0])
        weight = weight - gradient * lr

    print(iterr)


    #OUTPUT FROM LEARNED WEIGHTS
    pred_prob = softmax(np.dot(test_X,np.transpose(weight)))
    pred_classindex = np.argmax(np.dot(test_X,np.transpose(weight)),axis = 1)
    pred_class = []


    for element in pred_classindex:
        if element == 0:
            pred_class.append('not_recom')
            #print('not_recom')
        elif element == 1:
            pred_class.append('recommend')
            #print('recommend')
        elif element == 2:
            pred_class.append('very_recom')
            #print('very_recom')
        elif element == 3:
            pred_class.append('priority')
            #print('priority')
        elif element == 4:
            pred_class.append('spec_prior')
            #print('spec_prior')
        else:
            print('dang')

    print(len(pred_class))

    with open(outputfile, 'w', newline="") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(pred_class)

    np.savetxt(weightfile, np.transpose(weight), delimiter=",")
    print(lrs, ' -> s',lr,' -> ', max_iter)

elif lrs == '2':
    loss_prev = 10000
    for iterr in range(max_iter):
        a = np.dot(par,np.transpose(weight))
        error = np.subtract(softmax(a), lab)
        gradient = np.dot(np.transpose(error),par)/(1 * par.shape[0])
        #shape_err -> N * k
        #shape_par -> N * (m + 1)
        #shape_grd -> k * (m + 1)
        #shape_wgt -> k * (m + 1)
        weight = weight - gradient * (lr/((iterr + 1)**0.5))
        loss = -np.sum(np.multiply(lab,np.log(softmax(a))))/(2 * par.shape[0])
        #print(loss)
        if loss_prev - loss < 0.000000001:
            break
        #print(loss_prev - loss)
        loss_prev = loss
        #print(gradient)
        #print(loss)

    print(iterr)

    #print(loss)

    #OUTPUT FROM LEARNED WEIGHTS
    pred_prob = softmax(np.dot(test_X,np.transpose(weight)))
    pred_classindex = np.argmax(pred_prob,axis = 1)
    pred_class = []

    for element in pred_classindex:
        if element == 0:
            pred_class.append('not_recom')
            #print('not_recom')
        elif element == 1:
            pred_class.append('recommend')
            #print('recommend')
        elif element == 2:
            pred_class.append('very_recom')
            #print('very_recom')
        elif element == 3:
            pred_class.append('priority')
            #print('priority')
        elif element == 4:
            pred_class.append('spec_prior')
            #print('spec_prior')

    #print(pred_class)

    with open(outputfile, 'w', newline="") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(pred_class)

    #print(weight)

    #with open(outputfile, 'w') as f:
    #    for item in pred_class:
    #        f.write(item)
    #        f.write(',')
    #np.savetxt(outputfile, np.transpose(pred_class), delimiter=",")
    np.savetxt(weightfile, np.transpose(weight), delimiter=",")
    print(lrs, ' -> s',lr,' -> ', max_iter)

#alpha = 0.1
#beta = 0.1

if lrs == '3':
    #lr = 2
    t = lr

    #print(alpha, ' ', beta)
    for iterr in range(max_iter):
        a = np.dot(par,np.transpose(weight))
        error = np.subtract(softmax(a), lab)
        gradient = np.dot(np.transpose(error),par)/(1 * par.shape[0])

        initial_loss = -(np.sum(np.multiply(lab,np.log(softmax(a)))) )/(1 * par.shape[0])

        #print(initial_loss)
        temp_weight = weight - gradient * t
        a_temp = np.dot(par,np.transpose(temp_weight))
        loss = -(np.sum(np.multiply(lab,np.log(softmax(a_temp)))))/(1 * par.shape[0])

        while loss - initial_loss >  -alpha * t * (np.linalg.norm(gradient,2) ** 2):
            t = beta * t
            temp_weight = weight - gradient * t
            a_temp = np.dot(par,np.transpose(temp_weight))
            loss = -(np.sum(np.multiply(lab,np.log(softmax(a_temp)))))/(1 * par.shape[0])



        weight = weight - gradient * t
        #loss = -(np.sum(np.multiply(lab,np.log(softmax(a)))))/(1 * par.shape[0])

        #print(iterr, ' -> ',t, ' -> ', loss)

    print(iterr)

    #print(loss)

    #OUTPUT FROM LEARNED WEIGHTS
    pred_prob = softmax(np.dot(test_X,np.transpose(weight)))
    pred_classindex = np.argmax(pred_prob,axis = 1)
    pred_class = []

    for element in pred_classindex:
        if element == 0:
            pred_class.append('not_recom')
            #print('not_recom')
        elif element == 1:
            pred_class.append('recommend')
            #print('recommend')
        elif element == 2:
            pred_class.append('very_recom')
            #print('very_recom')
        elif element == 3:
            pred_class.append('priority')
            #print('priority')
        elif element == 4:
            pred_class.append('spec_prior')
            #print('spec_prior')

    #print(pred_class)

    with open(outputfile, 'w', newline="") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(pred_class)

    #print(weight)

    #with open(outputfile, 'w') as f:
    #    for item in pred_class:
    #        f.write(item)
    #        f.write(',')
    #np.savetxt(outputfile, np.transpose(pred_class), delimiter=",")
    np.savetxt(weightfile, np.transpose(weight), delimiter=",")
    print(lr, ' -> ' ,alpha, ' -> ' ,beta)
#'''
