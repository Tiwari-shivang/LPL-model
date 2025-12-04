## accounts
[Id] ,[Description] ,[short_name] ,[account_status] ,[accounting_method] ,[address_line_1] ,[address_line_2] ,[address_line_3] ,[City] ,[State] ,[zip_code] ,[Country] ,[TotalMarketValue] ,[CashBalance] ,[AvailableCash] ,[TotalCashAvailable] ,[model_id] ,[CreatedAt] ,[UpdatedAt] ,[CreatedBy] ,[UpdatedBy] ,[IsDeleted]

## cash_transactions
[Id] ,[portfolio_account_id] ,[Amount] ,[transaction_type] ,[transaction_date] ,[Comments] ,[created_at] ,[updated_at] ,[IsDeleted] 

## Models
[Id] ,[Name] ,[Description] ,[IsActive] ,[IsDeleted] ,[CreatedDate] ,[LastModifiedDate] ,[CreatedBy] ,[UpdatedBy] 

## ModelSleeves
[Id] ,[ModelId] ,[SleeveId] ,[AllocationPercentage] ,[CreatedDate] ,[LastModifiedDate] ,[CreatedBy] ,[UpdatedBy] ,[IsDeleted] 

## OrderAllocations
[Id] ,[AllocationEstCost] ,[CurrentQuantity] ,[CurrPercent] ,[DeltaPercent] ,[EndPercent] ,[EndQuantity] ,[IsFractional] ,[ModAppPercent] ,[ModelPercent] ,[Quantity] ,[Tolerance] ,[OrderId] ,[AccountId] ,[IsDeleted] 

## Orders
[Id] ,[AccountName] ,[AccountingMethod] ,[ApprovalStatus] ,[Comment] ,[CreateDate] ,[EstCost] ,[FilledPrice] ,[FilledQuantity] ,[OrderId] ,[OrderType] ,[Quantity] ,[RequestId] ,[SettleDate] ,[State] ,[TradeDate] ,[Tran] ,[UpdatedAt] ,[SecurityId] ,[AccountId] ,[IsDeleted] 

## Securities
[Id] ,[Name] ,[Symbol] ,[CUSIP] ,[SecurityTypeId] ,[Currency] ,[Price] ,[LastPriceDate] ,[PreviousClosingPrice] ,[Rate] ,[Description] ,[IsActive] ,[IsTradeable] ,[CreatedAt] ,[UpdatedAt] ,[CreatedBy] ,[UpdatedBy] ,[IsDeleted] 

## SecurityTypes
[Id] ,[Name] ,[SecurityTypeCode] ,[PricingMultiplier] ,[ShareDecimal] ,[CFICode] ,[SecurityTypeDescription] ,[IsActive] ,[PriceDecimals] ,[HoldingPeriod] ,[StaleDataCheck] ,[StaleDataWindow] ,[CreatedAt] ,[UpdatedAt] ,[CreatedBy] ,[UpdatedBy] ,[IsDeleted]

## Sleeves
[Id] ,[Name] ,[Description] ,[IsActive] ,[IsDeleted] ,[CreatedDate] ,[LastModifiedDate] ,[CreatedBy] ,[UpdatedBy]

## SleeveSecurities
[Id] ,[SleeveId] ,[SecurityId] ,[AllocationPercentage] ,[CreatedDate] ,[LastModifiedDate] ,[IsDeleted]

## TaxLots
[Id] ,[OriginalPrice] ,[OriginalTradeDate] ,[Quantity] ,[ReservedQuantity] ,[SellPrice] ,[SoldQuantity] ,[TaxLotType] ,[OrderAllocationId] ,[AccountId] ,[SecurityId] ,[IsDeleted]