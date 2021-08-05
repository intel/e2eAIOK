# Load original json as CSV
import sys
from timeit import default_timer as timer

def process_meta(file):
    fi = open(file, "r")
    fo = open("item-info", "w")
    for line in fi:
        obj = eval(line)
        cat = obj["categories"][0][-1]
        print(obj["asin"] + "\t" + cat, file=fo)

def process_reviews(file):
    fi = open(file, "r")
    user_map = {}
    fo = open("reviews-info", "w")
    for line in fi:
        obj = eval(line)
        userID = obj["reviewerID"]
        itemID = obj["asin"]
        rating = obj["overall"]
        time = obj["unixReviewTime"]
        print(userID + "\t" + itemID + "\t" + str(rating) + "\t" + str(time), file=fo)


def main():
    local_prefix = "/home/vmagent/app/recdp/examples/python_tests/dien/raw_data/"
    t0 = timer()
    process_meta(local_prefix + 'meta_Books.json')
    process_reviews(local_prefix + 'reviews_Books.json')
    t1 = timer()

    print("Convert initial csv from json took %.3f secs" % (t1 - t0))

if __name__ == "__main__":
    main()