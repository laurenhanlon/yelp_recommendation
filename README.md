
### A Yelp-powered Restaurant Recommendation Program
##### Lauren Hanlon

This project was initially for a class project, but recently have taken an interest in it and am sprucing it up to make it more efficient. Originally it used a k-nearest means algorithm to make restaurant recommendations, but after researching algorithms further I've decided to implement some changes.

Changes I'm working on implementing:

- Implementing a k-nearest neighbors implementation (currently it's using k-nearest means)

_Note_ After researching K-means versus KNN, I think that a KNN approach might be more similar to how Yelp suggests restaurants to you. Using a clustering technique and grouping based on like factors might produce a better recommendation since you can use more qualitative information such as restaurant type verus a classification technique such as in K-means, which relies more heavily on more quantitative data. Then again, if you're just looking for a ranking then K-means might be the best call. I'll investigate this further.

- Giving ratings more weight to users with more reviews (as it's implemented now, it gives equal weight to all users)

- Allowing for a weighting feature in your search, such as giving more emphasis to food quality over price.

