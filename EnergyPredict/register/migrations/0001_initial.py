# Generated by Django 4.0.4 on 2022-08-03 21:07

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='session',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ip', models.CharField(default='', max_length=10)),
                ('judge', models.CharField(default='', max_length=4)),
                ('time', models.DateField(auto_now_add=True)),
            ],
        ),
    ]
