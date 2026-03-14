from django.db import models


class SearchLog(models.Model):
    """Log user searches for analytics."""
    category = models.CharField(max_length=100, blank=True, null=True)
    keyword = models.CharField(max_length=200, blank=True, null=True)
    budget = models.FloatField(null=True, blank=True)
    results_count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Search: {self.keyword or self.category} @ {self.timestamp}"

    class Meta:
        ordering = ['-timestamp']


class FavoriteProduct(models.Model):
    """Track favorited products (session-based)."""
    product_name = models.CharField(max_length=200)
    brand = models.CharField(max_length=100)
    session_key = models.CharField(max_length=40)
    added_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.product_name} by {self.brand}"


import random, string
from django.utils import timezone
from datetime import timedelta


class EcoUser(models.Model):
    """Lightweight user — email only, no password."""
    email      = models.EmailField(unique=True)
    name       = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.email


class OTPCode(models.Model):
    """One-time password for email login."""
    email      = models.EmailField()
    code       = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    used       = models.BooleanField(default=False)

    def is_valid(self):
        return not self.used and (timezone.now() - self.created_at) < timedelta(minutes=10)

    @classmethod
    def generate(cls, email):
        cls.objects.filter(email=email, used=False).update(used=True)
        code = ''.join(random.choices(string.digits, k=6))
        return cls.objects.create(email=email, code=code)

    def __str__(self):
        return f"{self.email} → {self.code}"


class UserSavedProduct(models.Model):
    """Products scanned/entered by users — these join the main dataset."""
    user          = models.ForeignKey(EcoUser, on_delete=models.CASCADE, related_name='products')
    name          = models.CharField(max_length=200)
    brand         = models.CharField(max_length=100, blank=True)
    category      = models.CharField(max_length=100, default='General')
    description   = models.TextField(blank=True)
    materials     = models.CharField(max_length=500, blank=True)
    cert          = models.CharField(max_length=300, blank=True)
    price         = models.FloatField(null=True, blank=True)
    barcode       = models.CharField(max_length=50, blank=True)

    # EcoMindNet predicted scores (stored after prediction)
    eco_score     = models.FloatField(null=True, blank=True)
    ethics_score  = models.FloatField(null=True, blank=True)
    carbon_level  = models.CharField(max_length=20, blank=True)
    overall_score = models.FloatField(null=True, blank=True)
    tags          = models.CharField(max_length=200, blank=True)  # comma-separated

    source        = models.CharField(max_length=50, default='manual')  # 'barcode' or 'manual'
    saved_at      = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} — {self.user.email}"

    class Meta:
        ordering = ['-saved_at']
