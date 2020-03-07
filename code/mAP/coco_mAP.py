import glob
import json
import os
import shutil


MINOVERLAPS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
coco_mAP = 0

for MINOVERLAP in MINOVERLAPS:
    tmp_files_path = "tmp_files"
    if not os.path.exists(tmp_files_path):
        os.makedirs(tmp_files_path)


    def voc_ap(rec, prec):
      rec.insert(0, 0.0)
      rec.append(1.0)
      mrec = rec[:]
      prec.insert(0, 0.0)
      prec.append(0.0)
      mpre = prec[:]
      for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
      i_list = []
      for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
          i_list.append(i)
      ap = 0.0
      for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
      return ap, mrec, mpre


    # 解析路径
    def file_lines_to_list(path):
        with open(path) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        return content


    # 得到真实框和类别
    ground_truth_files_list = glob.glob('ground-truth/*.txt')
    ground_truth_files_list.sort()
    if len(ground_truth_files_list) == 0:
        raise Exception("Error: No ground-truth files found!")


    gt_counter_per_class = {}
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        for line in lines_list:
            class_name, left, top, right, bottom = line.split()
            bbox = left + " " + top + " " + right + " " + bottom
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                gt_counter_per_class[class_name] = 1
        with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    # 得到预测框和类别
    predicted_files_list = glob.glob('predicted/*.txt')
    predicted_files_list.sort()
    if len(ground_truth_files_list) == 0:
        raise Exception("Error: No predicted files found!")

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in predicted_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            lines = file_lines_to_list(txt_file)
            for line in lines:
                tmp_class_name, confidence, left, top, right, bottom = line.split()
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    sum_AP = 0.0
    ap_dictionary = {}
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
        predictions_data = json.load(open(predictions_file))
        nd = len(predictions_data)
        tp = [0] * nd
        fp = [0] * nd
        for idx, prediction in enumerate(predictions_data):
            file_id = prediction["file_id"]
            gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            bb = [float(x) for x in prediction["bbox"].split()]
            for obj in ground_truth_data:
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
            min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if not bool(gt_match["used"]):
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[class_name] += 1
                    with open(gt_file, 'w') as f:
                        f.write(json.dumps(ground_truth_data))
                else:
                    fp[idx] = 1
            else:
                fp[idx] = 1

        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += ap
        # print(class_name + " AP = {0:.2f}%".format(ap*100))
    mAP = sum_AP / n_classes
    print(mAP)
    coco_mAP += mAP
    # print("mAP = {0:.2f}%".format(mAP * 100))
    shutil.rmtree(tmp_files_path)
print("coco_mAP = {0:.2f}%".format(coco_mAP * 10))
