def Val_optimal_threshold(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    Inhosp_Nos_val, EXAM_NOs_val,y_trues_val,y_scores_val = [], [], [], []
    y_probs_val=[]
    for Inhosp_No, EXAM_NO, x, y in loader:
        x, y = x.to(device), y.to(device)

        y_trues_val.extend(y.detach().cpu().numpy().tolist())
        Inhosp_Nos_val.extend(Inhosp_No)
        EXAM_NOs_val.extend(list(EXAM_NO))
        with torch.no_grad():
            logits = model(x)
            y_prob = logits[:, 1]
            y_prob = y_prob.detach().cpu().numpy()
            y_probs_val.extend(list(y_prob))

            y_scores_val.append(logits)


    y_trues_one_hot = torch.tensor([i for i in y_trues_val])

    y_trues_one_hot = to_categorical(y_trues_one_hot, 2)

    y_scores_val = torch.cat([i for i in y_scores_val], 0)

    y_scores_val = y_scores_val.cpu().numpy()

    auc = roc_auc_score(y_trues_one_hot, y_scores_val)
    print('val_auc_Val_optimal_threshold',auc)

   # print("y_probs",y_probs)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sensitivities=[]
    specificities = []
    accuracies=[]
    precisions=[]
    preds_all=[]
    for i, item in enumerate(thresholds):
        # Yuedens=0
        print('第{}th thresholds{}'.format(i, item))

        preds = []
        for j in y_probs_val:

            if j > item:
                preds.append(1)

            else:
                preds.append(0)
        preds_all.append(preds)
        confusion_matrix0 = confusion_matrix(y_trues_val, preds)
        print(confusion_matrix0)
        sensitivity = confusion_matrix0[1, 1] / (confusion_matrix0[1, 0] + confusion_matrix0[1, 1])
        specificity = confusion_matrix0[0, 0] / (confusion_matrix0[0, 0] + confusion_matrix0[0, 1])
        accuracy=(confusion_matrix0[0, 0] +confusion_matrix0[1, 1])/(confusion_matrix0[0, 0] +confusion_matrix0[1, 1]+
                                                                     confusion_matrix0[1, 0]+confusion_matrix0[0, 1])
        precision=confusion_matrix0[1, 1]/(confusion_matrix0[1, 1]+confusion_matrix0[0, 1])

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        accuracies.append(accuracy)
        precisions.append(precision)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    accuracies=np.array(accuracies)
    precisions = np.array(precisions)




    optimal_threshold,point,preds_need =Find_Optimal_Cutoff(sensitivities, specificities,preds_all,thresholds)

    print('optimal_threshold=', optimal_threshold)
    print("point=", point)

    return optimal_threshold,point,Inhosp_Nos_val,EXAM_NOs_val,y_trues_val,y_probs_val,preds_need