# Generated by Django 4.0.4 on 2022-08-03 21:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('login', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='userinfo',
            name='mailnum',
            field=models.CharField(default='', max_length=32),
        ),
        migrations.AddField(
            model_name='userinfo',
            name='phonenum',
            field=models.CharField(default='', max_length=32),
        ),
        migrations.AlterField(
            model_name='userinfo',
            name='password',
            field=models.CharField(default='', max_length=32),
        ),
        migrations.AlterField(
            model_name='userinfo',
            name='username',
            field=models.CharField(default='', max_length=32),
        ),
    ]
