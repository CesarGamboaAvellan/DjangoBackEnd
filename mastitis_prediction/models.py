from django.db import models

# Create your models here.

class Article(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=120)
    content = models.TextField()

    def __str__(self):
        return self.title