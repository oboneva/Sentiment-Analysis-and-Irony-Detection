import scrapy
from scrapy_splash import SplashRequest


class AmazonReviewSpider(scrapy.Spider):
    name = 'amazon_review'
    allowed_domains = ['amazon.com']

    base_urls = ["https://www.amazon.com/AutoExec-Wheelmate-Steering-Attachable-Surface/product-reviews/B00E1D1GY6/", "https://www.amazon.com/Images-SI-Uranium-Ore/product-reviews/B000796XXM/",
                 "https://www.amazon.com/Avoid-Huge-Ships-John-Trimmer/product-reviews/0870334336/", "https://www.amazon.com/Hutzler-3571-571-Banana-Slicer/product-reviews/B0047E0EII/"]

    start_urls = []

    for url in base_urls:
        first_page_url = f'{url}ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
        start_urls.append(first_page_url)

        for page in range(2, 50):
            next_page_url = f'{url}ref=cm_cr_arp_d_paging_btm_next_{page}?ie=UTF8&reviewerType=all_reviews&pageNumber={page}'
            start_urls.append(next_page_url)

    def start_requests(self):
        for url in self.start_urls:
            yield SplashRequest(url, self.parse, args={'wait': 0.5})

    def parse(self, response):
        data = response.css('#cm_cr-review_list')

        star_ratings = data.css('.review-rating')

        comments = data.css('.review-text')

        for count in range(1, len(star_ratings)):
            yield {'stars': "".join(star_ratings[count].xpath('.//text()').extract()),
                   'comment': "".join(comments[count].xpath('.//text()').extract())}
