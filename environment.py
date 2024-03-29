"--------- ENVIRONMENT --------- "

import numpy as np


class Environment:
    """
    This class represents the Environment
    """
    
    def __init__(self):
        self.prices = np.linspace(22,40,10)
        self.bids = np.linspace(0.1, 1, num=10)
        product_price = 22
        self.product_price = product_price
        self.customer_class_1 = Customer(a_new_users=-5, b_new_users=1.5, c_new_users=0.9, d_new_users=100, var_new_users=2,
                                         a_cost_per_click=10, b_cost_per_click=1, min_cost_per_click=0.95,
                                         a_conversion_rate=-3.7, b_conversion_rate=-10, c_conversion_rate=0.1, d_conversion_rate=34.8, e_conversion_rate=-0.2,
                                         price_min=product_price,
                                         mean_n_times_comeback=5, dev_n_times_comeback=0.2)
        self.customer_class_2 = Customer(a_new_users=-0.8, b_new_users=0.3, c_new_users=-6, d_new_users=71.2, var_new_users=2,
                                         a_cost_per_click=5, b_cost_per_click=5, min_cost_per_click=0.9,
                                         a_conversion_rate=-4.8, b_conversion_rate=-3, c_conversion_rate=0.1, d_conversion_rate=21, e_conversion_rate=-19.5,
                                         price_min=product_price,
                                         mean_n_times_comeback=10, dev_n_times_comeback=0.2)
        self.customer_class_3 = Customer(a_new_users=-4.5, b_new_users=-1, c_new_users=1, d_new_users=50, var_new_users=2,
                                         a_cost_per_click=1, b_cost_per_click=10, min_cost_per_click=0.85,
                                         a_conversion_rate=-3.05, b_conversion_rate=1.2, c_conversion_rate=84, d_conversion_rate=20.1, e_conversion_rate=-13,
                                         price_min=product_price,
                                         mean_n_times_comeback=5, dev_n_times_comeback=0.2)

        self.customer_classes = [self.customer_class_1, self.customer_class_2, self.customer_class_3]

    def get_margin(self, selling_price):
        """
        Given a selling price this function returns the margin 
        :param selling: the proposed selling price
        :return: the margin 
        """
        selling_price = ( selling_price - self.product_price ) / 2
        return selling_price

    def get_next_30_days(self, n_customers, price, chosen_class):
        """
        Given the number of customers for each class and the price,
        this function returns the earning margin of the next 30 days
        :param n_customers: number of customers
        :param price: the proposed price
        :param chosen_class: the proposed class
        :return: a list of the next 30 days of margin
        """
        
        customer_class = self.customer_classes[chosen_class - 1]
        earning_margin = self.get_margin(price)

        next_30_days = [0] * 30

        for i in range(n_customers):
            this_customer_next_30_days = customer_class.get_comeback_days()
            for j in this_customer_next_30_days:
                next_30_days[j-1] += earning_margin

        return next_30_days

    def get_all_new_users_daily(self,bid):
        """
        Given a bid this function returns the number of new users for each class
        :param bid: the proposed bid
        :return: a triplet with (n_users_class1, n_users_class2, n_users_class3)
        """
        new_user_1 = self.get_new_users_daily(bid, 1)
        new_user_2 = self.get_new_users_daily(bid, 2)
        new_user_3 = self.get_new_users_daily(bid, 3)

        return new_user_1, new_user_2, new_user_3

    def get_mean_cost_per_click(self, bid, chosen_class):
        """
        Given a bid, this function returns the mean cost of every click for the given customer class
        :param bid: the proposed class
        :param chosen_class: the requested customer class
        :return: the cost of every click for the given customer class
        """
        return self.customer_classes[chosen_class - 1].mean_cost_per_click(bid)


    def get_cost_per_click(self, bid, chosen_class):
        """
        Given a bid, this function returns the cost of every click for the given customer class
        :param bid: the proposed class
        :param chosen_class: the requested customer class
        :return: the cost of every click for the given customer class
        """
        return self.customer_classes[chosen_class - 1].cost_per_click_daily(bid)

    def get_all_cost_per_click(self,bid):
        """
        Given a bid, this function returns the cost of every click for every customer in each category
        :param bid: the proposed bid
        :return: a triplet containing the cost of each click for evey customer class
        """
        cost1 = self.get_cost_per_click(bid, 1)
        cost2 = self.get_cost_per_click(bid, 2)
        cost3 = self.get_cost_per_click(bid, 3)
        return cost1, cost2, cost3

    def get_conversion_rate(self, price, chosen_class):
        """
        Given a class, returns it's conversion rate
        :param price: price for which calculate the conversion rate
        :param chosen_class: the class for which calculate the conversion rate
        :return: a percentage that indicates the conversion rate
        """
        return self.customer_classes[chosen_class - 1].conversion_rate(price)

    def buy(self, price, chosen_class):
        """
        This function tells if the user will buy or not the item
        :param price: price of selling
        :param chosen_class: class the user belongs
        :return: 1 if buys, 0 if not
        """
        return self.customer_classes[chosen_class - 1].buy(price)

    def n_time_comeback(self, chosen_class):
        """
        This function returns a stocastic number of how many times the user will come back
        :param chosen_class: class the user belongs
        :return: number of times the user will come back in the following 30 days
        """
        return self.customer_classes[chosen_class].n_times_comeback()

    def get_new_users_daily(self, bid, chosen_class):
        return self.customer_classes[chosen_class - 1].new_users_daily_clicks(bid)

    def get_mean_new_users_daily(self,bid,chosen_class):
        return self.customer_classes[chosen_class - 1].new_users_daily_clicks_mean(bid)

    def get_mean_n_times_comeback(self, chosen_class):
        return self.customer_classes[chosen_class - 1].get_mean_n_times_comeback()

    def get_comeback_days(self, chosen_class):
        return self.customer_classes[chosen_class - 1].get_comeback_days()



class Customer:
    """ This class represents a customer class.
    Three items of this class will be instantiated: C1, C2, C3.

    Attributes:
        seed    Example attribute
    """

    def __init__(self,
                 a_new_users, b_new_users, c_new_users, d_new_users, var_new_users,
                 a_cost_per_click, b_cost_per_click, min_cost_per_click,
                 a_conversion_rate, b_conversion_rate, c_conversion_rate, d_conversion_rate, e_conversion_rate,
                 price_min,
                 mean_n_times_comeback, dev_n_times_comeback):
        """

        :param a_new_users: 1_bid coeff
        :param b_new_users: 2_bid coeff
        :param c_new_users: 3_bid coeff
        :param d_new_users: 4_bid coeff
        :param var_new_users: variance of new users
        :param a_cost_per_click: const_per_click_coeff_1
        :param b_cost_per_click: const_per_click_coeff_2
        :param min_cost_per_click: const_per_click_coeff_3
        :param a_conversion_rate: coeff_1 conversion_rate 
        :param b_conversion_rate: coeff_2 conversion_rate 
        :param c_conversion_rate: coeff_3 conversion_rate 
        :param d_conversion_rate: coeff_4 conversion_rate 
        :param e_conversion_rate: coeff_5 conversion_rate 
        :param mean_n_times_comeback: mean of the normal distribution
        :param dev_n_times_comeback: standard deviation of the normal distribution
        """
        self.a_new_users = a_new_users
        self.b_new_users = b_new_users
        self.c_new_users = c_new_users
        self.d_new_users = d_new_users
        self.var_new_users = var_new_users

        self.a_cost_per_click = a_cost_per_click
        self.b_cost_per_click = b_cost_per_click
        self.min_cost_per_click = min_cost_per_click

        self.a_conversion_rate = a_conversion_rate
        self.b_conversion_rate = b_conversion_rate
        self.c_conversion_rate = c_conversion_rate
        self.d_conversion_rate = d_conversion_rate
        self.e_conversion_rate = e_conversion_rate
        self.price_min = price_min

        self.mean_n_times_comeback = mean_n_times_comeback
        self.dev_n_times_comeback = dev_n_times_comeback

    def new_users_daily_clicks_mean(self, bid):
        a = self.a_new_users
        b = self.b_new_users
        c = self.c_new_users
        d = self.d_new_users
        return d * (1.0 - np.exp(a * bid + b * bid ** 2 + c * bid ** 3))

    def new_users_daily_clicks(self, bid):
        """ Customer's characteristic n.1.
        This method returns a stochastic number of daily clicks of new users (i.e., that have never clicked before these
        ads) as a function depending on the bid.
        :param bid: The seller's bid
        :return: Number of new users that clicks on the ads in a day
        """
        # Functions that assign the number of clicks to a given budget
        # They are monotone increasing in [0,1]
        mean = self.new_users_daily_clicks_mean(bid)
        var = self.var_new_users
        return round(np.random.normal(loc=mean, scale=var))

    def mean_cost_per_click(self, bid):
        """
        This method returns a daily mean cost per click as a function of the bid.
        The seller pays more for this class depending on his bid.
        :param bid: The seller's bid
        :return: The mean cost that will be due for the customer of this customer with the given bid
        """

        return ( ( self.a_cost_per_click/(self.a_cost_per_click + self.b_cost_per_click) ) * (1-self.min_cost_per_click) + self.min_cost_per_click ) * bid

    def cost_per_click_daily(self, bid):
        """ Customer's characteristic n. 2.
        This method returns a daily stochastic cost per click as a function of the bid.
        The seller pays more for this class depending on his bid. What the seller pays is not what he bids: here we must
        model the second price auction without having the other bids.
        :param bid: The seller's bid
        :return: The cost that will be due for the customer of this customer with the given bid
        """
        #it is a beta normalized on [min_cost_per_click,1]

        return ( np.random.beta(self.a_cost_per_click,self.b_cost_per_click) * (1-self.min_cost_per_click) + self.min_cost_per_click ) * bid

        #return self.a_cost_per_click * bid

    def conversion_rate(self, price):
        """ Customer's characteristic n. 3.
        This method is a conversion rate function. It provids the probability that a user will buy the item given a
        price.
        :param price: The item's price
        :return: The probability that a user will buy the item given a price
        """

        price = ( price - 20 ) / 2

        a = self.a_conversion_rate
        b = self.b_conversion_rate
        c = self.c_conversion_rate
        d = self.d_conversion_rate
        e = self.e_conversion_rate
        # price_min = self.price_min
        # Probabilities of conversion given a price
        return c * np.exp ( a * ( price - e) ** (1/ (2 * b) ) ) * (d - 2*price) ** (3/2)

    def buy(self, price):
        """
        This method returns if the customer will buy or not the item at the given price
        :param price: given price
        :return: 1 if the customer will buy they item. 0 otherwise.
        """
        return np.random.binomial(1, self.conversion_rate(price))

    def n_times_comeback(self):
        """ Customer's characteristic n. 4.
        This method computes the distribution probability over the number of times the user will come back to the
        ecommerce website to buy that item by 30 days after the first purchase (and simulate such visits in future).
        :return: The probability that the user of this class will come back the given number of times
        """
        # OR Poisson or Gaussian
        loc = self.get_mean_n_times_comeback()
        scale = self.dev_n_times_comeback
        return int(np.around(np.random.normal(loc=loc, scale=scale, size=1)))

    def get_comeback_days(self):
        """
        This method returns the days when a customer will comeback in the following 30 days after the first purchase
        :return: the list of days when the customer will come back after the first purchase
        """
        n_comebacks = self.n_times_comeback()

        comebacks = []
        if n_comebacks == 0:
            return comebacks
        else:
            period = 30 / float(n_comebacks)
            for i in range(1, n_comebacks + 1):
                mean = i*period
                return_day = np.around(np.random.normal(loc=mean, scale=2, size=1))
                comebacks.append(int((max(min(return_day, 30), 1))))

            return comebacks

    def get_mean_n_times_comeback(self):
        """
        This method returns the mean times a customer will comeback in the following 30 days after the first purchase
        :return: mean times comebacks
        """
        return self.mean_n_times_comeback
