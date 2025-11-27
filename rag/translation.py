from transformers import MT5ForConditionalGeneration, MT5Tokenizer

class EnglishToPersian:
    def __init__(self, model_name="persiannlp/mt5-small-parsinlu-translation_en_fa"):
        self.model_name = model_name
        self.tokenizer = MT5Tokenizer.from_pretrained(self.model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name)

    def translate(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
        res = self.model.generate(input_ids, **generator_args)
        output = self.tokenizer.batch_decode(res, skip_special_tokens=True)

        return output[0]

class PersianToEnglish:
    def __init__(self, model_name="persiannlp/mt5-small-parsinlu-opus-translation_fa_en"):
        self.model_name = model_name
        self.tokenizer = MT5Tokenizer.from_pretrained(self.model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name)

    def translate(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
        res = self.model.generate(input_ids, **generator_args)
        output = self.tokenizer.batch_decode(res, skip_special_tokens=True)

        return output[0]        
