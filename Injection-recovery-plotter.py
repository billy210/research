################################
## Injection Recovery Plotter ##
################################

def inject_plotting(outtable, stellar_radius, Tmag):
    
    output_table = outtable

    radlist1=np.linspace(0.6,1.9,14)
    radlist2=np.linspace(0.7,2,14)

    detectionmatrix = np.empty((1,11))

    for r1,r2 in zip(radlist1,radlist2):
        p1 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(0, 1))]
        recoverpercentp1 = (p1['recover?'].sum())/len(p1.index)
        #print(recoverpercentp1)

        p2 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(1, 2))]
        recoverpercentp2 = (p2['recover?'].sum())/len(p2.index)
        #print(recoverpercentp2)

        p3 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(2, 3))]
        recoverpercentp3 = (p3['recover?'].sum())/len(p3.index)
        #print(recoverpercentp3)

        p4 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(3, 4))]
        recoverpercentp4 = (p4['recover?'].sum())/len(p4.index)
        #print(recoverpercentp4)

        p5 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(4, 5))]
        recoverpercentp5 = (p5['recover?'].sum())/len(p5.index)
        #print(recoverpercentp5)

        p6 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(5, 6))]
        recoverpercentp6 = (p6['recover?'].sum())/len(p6.index)
        #print(recoverpercentp6)

        p7 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(6, 7))]
        recoverpercentp7 = (p7['recover?'].sum())/len(p7.index)
        #print(recoverpercentp7)

        p8 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(7, 8))]
        recoverpercentp8 = (p8['recover?'].sum())/len(p8.index)
        #print(recoverpercentp8)

        p9 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(8, 9))]
        recoverpercentp9 = (p9['recover?'].sum())/len(p9.index)
        #print(recoverpercentp9)
        #print('           ')


        p10 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(9, 10))]
        recoverpercentp10 = (p10['recover?'].sum())/len(p10.index)
        #print(recoverpercentp1)

        p11 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(10, 11))]
        recoverpercentp11 = (p11['recover?'].sum())/len(p11.index)
        #print(recoverpercentp2)
    

        matrixgrid = np.array([[recoverpercentp1,recoverpercentp2,recoverpercentp3,recoverpercentp4,recoverpercentp5,
                                recoverpercentp6,recoverpercentp7,recoverpercentp8,recoverpercentp9,recoverpercentp10,
                                recoverpercentp11]])
        detectionmatrix = np.concatenate((detectionmatrix, matrixgrid), axis=0)

    detectionmatrix = np.delete(detectionmatrix, obj=0, axis=0)



    fig, ax = plt.subplots(figsize=(10, 10))
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(detectionmatrix, cmap='plasma')

    for (i, j), z in np.ndenumerate(detectionmatrix):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
        ax.margins(x=0)

    ax.set_xticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5])
    ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11])

    ax.set_yticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5])
    ax.set_yticklabels([0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])

    plt.gca().invert_yaxis()

    plt.tick_params(labelbottom=True, labeltop=False)
    plt.title(f'Rad={stellar_radius}, Tmag={Tmag}')
    plt.xlabel('Period [days]')
    plt.ylabel('R_planet [R_Jup]')

    #plt.xscale('log')

    #ax.scatter(output_table['Pinject'],output_table['Rplanet'])

    plt.show()
