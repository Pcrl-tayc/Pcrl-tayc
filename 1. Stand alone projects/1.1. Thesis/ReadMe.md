# Dissertation project
## FinTech and BigTech credit: a machine learning approach
### Vrije Universiteit Amsterdam
Leading tech companies have penetrated the credit market business, developing lending platforms and nondisclosed rating models, with world-spanning infrastructure (Elsinger et al., 2018), and raising concerns regarding their potential systemic impact. Along with other forms of credit, such new types of lending have the potential to support economic growth, foster financial inclusion, with innovations reaching millions of users in a remarkably short period, enhance efficiency, and expand the range of available funding sources for households and firms. However, there is still an ongoing regulatory debate with regards to the possible risks to the macroeconomy and financial system that they engender, given for instance the hyper-scalability and wide access to customers´ data that BigTech firms have, and the need to test many of the innovations over a full financial cycle. Hence, decisions made at this early stage can set important precedents. (Cornelli et al., 2020; FSB, 2017)

As some authors point out (Greene, 2000; Wooldridge, 2010; Brooks, 2019; Gujarati, 2021) traditional econometrics often focuses on the estimation of a parameter of a statistical model describing the distribution of a given set of variables, and the focus is on the quality of that estimation based on confidence intervals and standard errors. While the ability to construct valid confidence intervals over large samples remains important in many cases, approaches not yet capable of providing them should not be discarded altogether if these approaches have other advantages. This is especially true if these methods have been shown to outperform out-of-sample predictive power in specific data sets. (Athey & Imbens, 2019)

Leveraging on the ability of machine learning systems to learn from the data, this paper employs the variables contributed by the previous studies, and expands on them by including others specific to this study, to develop a model capable of effectively predicting alternative credit volumes. The rationale for employing this new methodology is based on the emerging idea of moving away from exclusive dependence on data models and using newly available tools as an alternative to data modeling to solve problems more accurately. (Breiman, 2001)

## Data Employed
### Dependent Variables

Total Alternative Credit- Defined as the sum of FinTech credit and BigTech credit provided in the ith economy at time t.

### Independent Variables
 
   (World Bank)- GDP/capita (current USD). GDP per capita is gross domestic product divided by midyear population.
   
   (World_Bank)- Income Level- PDF obtained from the WorldBank. Converted into dummy variables in the following way:
   
   (GitHub)-Region & Sub-region. Text file obtained from one repo on github.

   (World Bank)- GDP growth (% per annum). Annual percentage growth rate of GDP per capita based on constant local currency. Aggregates are based on constant 2010 U.S. dollars.
   
   (World Bank)- Consumer Price Index (2010=100).Consumer price index reflects changes in the cost to the average consumer of acquiring a basket of goods and services that may be fixed or changed at specified intervals, such as yearly.

   (World Bank)-Commercial Bank Branches (per 100k adults). Number of branches of commercial banks for every 100,000 adults in the reporting country. It is calculated as (number of institutions + number of branches)*100,000/adult population in the reporting country. Access to finance can expand opportunities for all with higher levels of access and use of banking services associated with lower financing obstacles for people and businesses. There are several aspects of access to financial services: availability, cost, and quality of services.

   (World Bank)-Individuals using the Internet (% of population)(IT.NET.USER.ZS). Internet users are individuals who have used the Internet (from any location) in the last 3 months. The Internet can be used via a computer, mobile phone, personal digital assistant, games machine, digital TV etc.
   
   (World Bank)- Lerner Index. A measure of market power in the banking market. It is defined as the difference between output prices and marginal costs (relative to prices). Prices are calculated as total bank revenue over assets, whereas marginal costs are obtained from an estimated translog cost function with respect to output. (Imputed using iterative imputer since it had missing values for the period considered)
   
   (World Bank)-Research & Development-Gross domestic expenditures on research and development (R&D), expressed as a percent of GDP. They include both capital and current expenditures in the four main sectors: Business enterprise, Government, Higher education and Private non-profit. R&D covers basic research, applied research, and experimental development.
   
   (IMF)-Financial Development Index . Is a relative ranking of countries on the depth, access, and efficiency of their financial institutions and financial markets.
   
   (World Bank)- Starting a Business: Score. The score for starting a business is the simple average of the scores for each of the component indicators: the procedures, time and cost for an entrepreneur to start and formally operate a business, as well as the paid-in minimum capital requirement.
   
   (World Bank)-Starting a Business: Score-Time(days)- Time required to start a business (days). Time required to start a business is the number of calendar days needed to complete the procedures to legally operate a business. If a procedure can be speeded up at additional cost, the fastest procedure, independent of cost, is chosen.
   
   (World Bank)-Starting a Business: Score-in-Minimum Capital(% of income per capita). The score for the paid-in minimum capital requirement benchmarks economies with respect to the regulatory best practice on the indicator. The score ranges from 0 to 100, where 0 represents the worst regulatory performance and 100 the best regulatory performance.
   
   (World Bank)-Starting a Business: Cost of business start-up procedures (% of GNI per capita). Cost to register a business is normalized by presenting it as a percentage of gross national income (GNI) per capita.
   
   (World Bank)-Extent of disclosure Index (0-10). Disclosure index measures the extent to which investors are protected through disclosure of ownership and financial information. The index ranges from 0 to 10, with higher values indicating more disclosure.
   
   (World Bank)-Enforcement fees (% of claim). The enforcement fees are all costs that plaintiff must advance to enforce the judgment through a public sale of defendant’s movable assets, regardless of the final cost borne by plaintiff.
   
   (World Bank)-Enforcement of judgment (days).The time for enforcement of judgment captures the time from the moment the time to appeal has elapsed until the money is recovered by the winning party.
   
   (World Bank)- Cost (% of claim). The cost to enforce contracts is recorded as a percentage of the claim value, assumed to be equivalent to 200% of income per capita or USD 5,000, whichever is greater. Three types of costs are recorded: average attorney fees, court costs and enforcement costs. Bribes are not taken into account.
   
   (World Bank)-Bank Regulatory Capital To Risk-Weighted Assets (%). The capital-to-risk-weighted assets ratio for a bank is usually expressed as a percentage. The current minimum requirement of the capital-to-risk weighted assets ratio, under Basel III, is 10.5%, including the conservation buffer.
   
   (World Bank)- Provisions to non-performing loans(%). 
   
   (World Bank)-Loans from non-resident banks to GDP(%)-Ratio of outstanding offshore bank loans to GDP. An offshore bank is a bank located outside the country of residence of the depositor, typically in a low tax jurisdiction (or tax haven) that provides financial and legal advantages. Offshore bank loan data from BIS Statistical Appendix Table 7A: External loans and deposits of reporting banks vis-à-vis all sectors.
   
   (World Bank)- Corporate Bond issuance volume to GDP (%).
   
   (World Bank)- Total Factoring Volumen to GDP (%).
   
   (World Bank)- Global Leasing Volume to GDP (%)
   
   (World Bank)- Domestic Credit provided by the Financial sector as % of GDP.Domestic credit provided by the financial sector includes all credit to various sectors on a gross basis, with the exception of credit to the central government, which is net. The financial sector includes monetary authorities and deposit money banks, as well as other financial corporations where data are available (including corporations that do not accept transferable deposits but do incur such liabilities as time and savings deposits). Examples of other financial corporations are finance and leasing companies, money lenders, insurance corporations, pension funds, and foreign exchange companies.
   
   (World Bank)- Banking Crisis Dummy (1=banking crisis, 0=none)
   
   (World Bank)-Stocks Traded Total as (%GDP).The value of shares traded is the total number of shares traded, both domestic and foreign, multiplied by their respective matching prices. Figures are single counted (only one side of the transaction is considered). Companies admitted to listing and admitted to trading are included in the data. Data are end of year values.
   
   (World Bank)- Mobile cellular subscriptions (per 100 people). Data on mobile cellular subscribers are derived using administrative data that countries (usually the regulatory telecommunication authority or the Ministry in charge of telecommunications) regularly, and at least annually, collect from telecommunications operators.Mobile cellular subscriptions (per 100 people) indicator is derived by all mobile subscriptions divided by the country's population and multiplied by 100.



   
