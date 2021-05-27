from ret_fundus import *

datasets = "./datasets/Accepted images"
if not os.path.exists("./results"): os.makedirs("./results")

_, _, files = next(os.walk(datasets))

for file in files:
    try:
        fhand = open(f"./results/datasets_results.txt", 'r')
    except OSError:
        print("Results file cannot be read!")
        quit()
    
    name = file.split('.')[0]

    flag = False

    for line in fhand:
        if name in line:
            flag = True

    fhand.close()

    if flag: continue

    print(f"Evaluating {name}.")
    
    retina = retinal_fundus(f"{datasets}/{file}")
    comps = cv.circle(retina.retina, retina.Optic_Disc['Center'], retina.Optic_Disc['Radius'], (0xFF,0,0), 3)
    comps = cv.circle(comps, retina.Macula['Center'], retina.Macula['Radius'], (0,0xFF,0), 3)

    figure, axs = plt.subplots(1, 2)
    axs[0].imshow(retina.Vascular_tree['Mask'], cmap = 'gray')
    axs[0].set_title(f"{name} vascular tree")
    axs[1].imshow(comps)
    axs[1].set_title(f"{name} Optic Disc & Macula")
    (ax.set_axis_off() for ax in axs)

    plt.savefig(f"./results/{name}_results.png", dpi=500)
    plt.clf()

    try:
        fhand = open(f"./results/datasets_results.txt", 'a')
    except OSError:
        print("Results file cannot be read!")
        quit()

    result = f"""
    {name} results:
        Optic Disc -> Center: {retina.Optic_Disc['Center']}; Radius: {retina.Optic_Disc['Radius']}
        Macula -> Center: {retina.Macula['Center']}; Radius: {retina.Macula['Radius']}
    """

    print(f"{name} has been successfuly evaluated!")

    fhand.write(result)
    fhand.close()

print("All files found in the datasets have been evaluated! Their results are found in document 'datasets_results.txt'.")
