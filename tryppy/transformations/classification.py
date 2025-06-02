from collections import defaultdict, Counter


class Classification:
    def __init__(self, file_handler):
        self.file_handler = file_handler

    def prepare_data(self, dataframe):
        ids = dataframe[["crop_id", "orig_image"]]
        df = dataframe.drop(columns=["crop_id", "orig_image"])
        return df, ids

    def pred_to_dict(self, pred):
        nested_dict = defaultdict(dict)
        for _, row in pred.iterrows():
            image_id = row['orig_image']
            crop_id = row['crop_id']
            prediction = row['predicted']
            nested_dict[image_id][crop_id] = prediction
        return dict(nested_dict)

    def predict(self, dataframe, confidence=False):
        input_data, ids = self.prepare_data(dataframe)

        rf, label_encoder = self.file_handler.load_rf()
        pred = rf.predict(input_data)

        results = ids.copy()
        results['predicted'] = label_encoder.inverse_transform(pred)
        return results

    def run(self, dataframe):
        prediction = self.predict(dataframe)
        prediction_dict = self.pred_to_dict(prediction)
        class_count = self.count_classes(prediction_dict)
        return prediction_dict, class_count

    def count_classes(self, prediction_dict):
        class_count ={
            image_id: dict(Counter(crop_predictions.values()))
            for image_id, crop_predictions in prediction_dict.items()
        }
        return class_count
