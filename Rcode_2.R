suppressMessages(library(mlbench))      
suppressMessages(library(caret))        
suppressMessages(library(dplyr))        
suppressMessages(library(e1071))        
suppressMessages(library(tidyr))        
suppressMessages(library(knitr))        
suppressMessages(library(GGally))

data(Soybean)

str(Soybean)
total.count<-Soybean%>%
  group_by(Class)%>%
  summarise(total=n()) 
temp<-sapply(Soybean[,2:36],function(fact.num){table(Soybean$Class,fact.num)})
for(i in 1:35){
  temp2<-data.frame(temp[[i]]) 
  temp2$name<-names(temp)[i]   
  ifelse(i==1,count.byclass<-temp2,count.byclass<-rbind(count.byclass,temp2)) 
}
names(count.byclass)<-c("Class","factor.value","count","factor") 
count.byclass<-left_join(count.byclass,total.count,by="Class") 
count.byclass$pct<-count.byclass$count/count.byclass$total 
count.byclass<-tbl_df(count.byclass)
rm(temp,temp2,i)
for(i in 1:35){
  readline(prompt="Hit return to see next plot>> ")
  fact.nam<-names(Soybean)[i+1]
  print(ggplot(subset(count.byclass,(factor==fact.nam)),aes(x=Class,y=pct,fill=factor.value))+
          geom_bar(stat="identity",position="stack")+
          theme(axis.text.x=element_text(angle=90))+
          labs(ylab("Percent of Total (by Class)")) +
          ggtitle(paste("Series Chart ",i,": Percent of Observations by Factor Value for Explanatory Variable [",
                        fact.nam,"] \n (by Class)",sep=""))+
          theme(plot.title=element_text(size=10))+
          annotate(geom="text",label="Failure to sum to 100% is due to NAs",x=15,y=.1,size=3,colour="grey25")
  )
}
# Begin of additions by Navya
zero_cols = nearZeroVar(Soybean)
colnames(Soybean)[zero_cols]
Soybean = Soybean[,-zero_cols]
Soybean
# End of additions by Navya

rm(fact.nam,i)
isna.byxvar<-data.frame(sapply(Soybean[,2:36],function(isna){sum(is.na(isna))}))
isna.byxvar$x.var<-rownames(isna.byxvar)
rownames(isna.byxvar)<-NULL
names(isna.byxvar)<-c("na.count","x.var")
isna.byxvar

ggplot(isna.byxvar,aes(x=reorder(x.var,-na.count),y=na.count))+
  geom_bar(stat="identity",fill="light blue",colour="black")+
  theme(axis.text.x=element_text(angle=90))+
  xlab("Explanatory Variable")+
  ylab("Number of NAs")+
  ggtitle("Number of NAs by Explanatory Variable")+
  theme(plot.title=element_text(size=10))

isna.byclass<-Soybean%>%
  group_by(Class)%>%
  do(data.frame(sum(is.na(.))))
names(isna.byclass)<-c("Class","na.count")
isna.byclass

ggplot(isna.byclass,aes(x=reorder(Class,-na.count),y=na.count))+
  geom_bar(stat="identity",fill="light blue",colour="black")+
  theme(axis.text.x=element_text(angle=90))+
  xlab("Class of Disease")+
  ylab("Number of NAs")+
  ggtitle("Number of NAs by Class of Disease")+
  theme(plot.title=element_text(size=10))

isna.all<-Soybean%>%
  group_by(Class)%>%
  do(data.frame(apply(.,2,function(x){length(which(is.na(x)))})))
isna.all$x.var<-names(Soybean)
names(isna.all)<-c("Class","na.count","var")
isna.all<-filter(isna.all,var!="Class")

isna.table<-spread(isna.all,Class,na.count)

ggplot(isna.all,aes(x=Class,y=var,size=na.count,colour=na.count))+
  geom_point()+
  theme_bw()+
  theme(axis.text.x=element_text(angle=90))+
  ggtitle("Number of NAs \n (by Class and Variable)")+
  theme(plot.title=element_text(size=8))+
  ylab("Explanatory Variable")+
  scale_colour_gradientn(colours=topo.colors(50))
  